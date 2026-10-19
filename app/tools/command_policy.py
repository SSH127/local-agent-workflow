"""Argument-level command policy.

The original policy allowlisted *program names* only. That made
``python -m pytest _recover.py -q -s`` look safe (``pytest`` is allowlisted)
while pytest imported and executed an arbitrary script. The same hole existed
for every wrapper that takes a code selector as a positional argument:
``python -m unittest evil``, ``python -m coverage run evil.py`` and
``python -m compileall <path outside workspace>``.

This module resolves the *real* entry point and validates the arguments that
select what code gets executed. It deliberately does NOT claim to be a sandbox:

* running a project's tests executes that project's code (that is what tests
  are for). The boundary enforced here is that only declared verification entry
  points run, only against paths inside the workspace, and never through
  plugin/config/import indirection that pulls code from an unvalidated place.
* anything whose behaviour cannot be statically bounded falls back to the
  existing approval gate (bound to run + exact args + cwd) instead of running
  silently.

Policy levels, applied per invocation:

``allowed``
    Provably a controlled verification run: real test-runner module, every
    explicit target is a test file/directory inside the workspace, and no
    option can pull in code from elsewhere.

``approval``
    Executes project code whose scope cannot be statically bounded (implicit
    whole-repo discovery, unknown options, package-manager scripts, ...).
    Requires an explicit, single-use user approval.

``reject``
    Pure bypass shapes with no legitimate use: a non-test script handed to
    pytest, plugin importing, alternate config files, ``--pyargs``, targets or
    write destinations outside the workspace, interpreter shadowing.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.tools.errors import SecurityError

# ---------------------------------------------------------------------------
# Decision model
# ---------------------------------------------------------------------------



_NETWORKY_TOKEN = re.compile(
    r"(?i)\b("
    r"curl|wget|invoke-webrequest|\biwr\b|"
    r"urllib|requests\.|httpx\.|aiohttp|"
    r"socket\.create_connection|socket\.connect|fetch\("
    r")\b"
)


def looks_networky(program: str, argv: list[str] | None = None) -> bool:
    """Best-effort heuristic: command likely opens the network.

    Used to force approval / refuse read-only subtasks. Not a firewall.
    """
    prog = (program or "").strip().lower()
    args = [str(a) for a in (argv or [])]
    if prog in {"curl", "wget"}:
        return True
    joined = " ".join(args)
    if prog in {"powershell", "pwsh"} and _NETWORKY_TOKEN.search(joined):
        return True
    if prog in {"python", "python3"} and _NETWORKY_TOKEN.search(joined):
        return True
    # npm install/ci/publish pull the network; keep approval path.
    if prog == "npm" and args and args[0].lower() in {"install", "ci", "publish", "update"}:
        return True
    if prog == "pip" and args and args[0].lower() in {"install", "download"}:
        return True
    return bool(prog == "uv" and args and args[0].lower() in {"pip", "add", "sync"})


@dataclass(frozen=True)
class CommandDecision:
    """Result of classifying one parameterized command."""

    program: str
    effective_program: str
    requires_approval: bool = False
    risk: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    # Declared write targets (workspace-relative POSIX; trailing "/" = the
    # command may write anywhere below that directory). Checked against the
    # running subtask's write_scope. Formatters/auto-fixers and dependency
    # installs declare concrete writes; cache/artifact writers leave this
    # empty and set ``cache_kind`` instead (redirected into a per-subtask
    # scratch + post-run escape check).
    writes: tuple[str, ...] = field(default_factory=tuple)
    # When set, ``run_project_command`` redirects known caches/artifacts into
    # ``.agent_scratch/<run>/<subtask>/`` and fails (restoring when possible)
    # if anything outside write_scope + scratch changed.
    # Values: pytest | mypy | npm | tsc | coverage | compileall | unittest
    cache_kind: str | None = None


def _rel_posix(path: Path, workspace: Path) -> str:
    try:
        rel = path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        rel = path.as_posix()
    return "" if rel == "." else rel


def _join(rel_dir: str, name: str) -> str:
    return f"{rel_dir}/{name}" if rel_dir else name


def dependency_writes(kind: str, cwd_dir: Path, workspace: Path) -> tuple[str, ...]:
    """What a dependency install/update may modify."""
    base = _rel_posix(cwd_dir, workspace)
    if kind == "npm":
        return (_join(base, "package.json"), _join(base, "package-lock.json"),
                _join(base, "node_modules") + "/")
    return (_join(base, "pyproject.toml"), _join(base, "uv.lock"),
            _join(base, "requirements.txt"), _join(base, ".venv") + "/")


# ---------------------------------------------------------------------------
# Shared policy tables
# ---------------------------------------------------------------------------

_ALLOWED_PROGRAMS: dict[str, set[str]] = {
    "pytest": set(),
    "ruff": {"check", "format"},
    "python": {"--version", "-m"},
    "npm": {"run", "test"},
    "npx": {"tsc"},
    "uv": {"run"},
    "mypy": set(),
}

_APPROVAL_SUBCOMMANDS: dict[str, set[str]] = {
    "npm": {"install", "ci", "uninstall", "update", "i"},
    "uv": {"pip", "add", "sync", "lock"},
}

# Modules reachable through ``python -m`` / ``uv run``. Being listed here is
# not by itself permission to execute: each module has its own validation.
_KNOWN_MODULES = {
    "pytest", "ruff", "mypy", "pip", "coverage", "compileall", "unittest",
}

# npm scripts treated as read-only pass/fail verification.
_VERIFYING_NPM_SCRIPTS = {"test", "build", "typecheck", "lint", "tsc"}

DENY_ARG_PATTERNS = (
    r"rm\s+-rf",
    r"-rf",
    r"del\s+/[sqf]",
    r">\s*\S",
    r">>",
    r"`",
    r"\$\(.*\)",
    r"\$\{.*\}",
    r"\|\s*(sh|bash|cmd|powershell)\b",
    r";\s*\w",
    r"&\s*\w",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bformat\s+[cdefgh]:",
    r"--no-verify",
    r"\n",
    r"\r",
)

# Environment variables that can redirect *which code* an interpreter imports
# or runs. A model may not supply them, and inherited values are dropped so a
# developer shell cannot silently inject plugins into an agent run.
BLOCKED_ENV_KEYS = {
    "pytest_addopts",
    "pytest_plugins",
    "pytest_debug",
    "pythonpath",
    "pythonhome",
    "pythonstartup",
    "pythoninspect",
    "pythonexecutable",
    "pythonuserbase",
    "node_options",
    "npm_config_script_shell",
}
# A model-supplied env may additionally not override executable resolution.
MODEL_BLOCKED_ENV_KEYS = BLOCKED_ENV_KEYS | {"path", "pathext"}

# ---------------------------------------------------------------------------
# pytest argument analysis
# ---------------------------------------------------------------------------

# A ``.py`` file is a recognised test module only by pytest's default naming
# convention. This is a *scoping* rule (which files may be executed as tests),
# not a safety proof: a test module still runs arbitrary project code.
_TEST_FILE_RE = re.compile(r"^(test_.*|.*_test)\.py$", re.IGNORECASE)

# Options that take no value.
_PYTEST_BOOL_LONG = {
    "--quiet", "--verbose", "--exitfirst", "--stepwise", "--stepwise-skip",
    "--collect-only", "--co", "--disable-warnings",
    "--continue-on-collection-errors", "--strict-markers", "--strict-config",
    "--runxfail", "--last-failed", "--failed-first", "--new-first",
    "--showlocals", "--full-trace", "--setup-show", "--setup-plan",
    "--fixtures", "--fixtures-per-test", "--markers", "--version", "--help",
    "--trace", "--no-header", "--no-summary", "--no-folders",
    "--cache-clear", "--keep-duplicates", "--sw", "--pdb",
}
# Options that take a value (``--opt value`` or ``--opt=value``).
_PYTEST_VALUE_LONG = {
    "--tb", "--capture", "--maxfail", "--durations", "--ignore",
    "--ignore-glob", "--deselect", "--import-mode", "--rootdir",
    "--confcutdir", "--basetemp", "--junitxml", "--junit-xml", "--log-file",
    "--log-level", "--log-format", "--log-date-format", "--log-cli-level",
    "--log-cli-format", "--log-cli-date-format", "--code-highlight", "--color",
    "--numprocesses", "--dist", "--assert", "--timeout",
    "--cov", "--cov-report", "--cov-config",
}
# Options that would load code/config from an unvalidated place, or re-target
# the run outside the workspace. There is no approval path: a legitimate task
# never needs them, and allowing them would re-open the bypass.
_PYTEST_HARD_DENY = {
    "--pyargs": "pytest 的 --pyargs 会按模块名导入代码，可指向任意已安装/工作区模块",
    "--config-file": "pytest 的 -c/--config-file 会从任意位置加载配置（addopts 可再注入插件）",
    "--override-ini": "pytest 的 -o/--override-ini 可改写 addopts/python_files 等收集规则",
    "--plugin": "pytest 的 --plugin 会导入并执行任意插件模块",
}
# Options whose value is a filesystem path that must stay inside the workspace.
_PYTEST_PATH_OPTIONS = {
    "--ignore", "--rootdir", "--confcutdir", "--basetemp", "--junitxml",
    "--junit-xml", "--log-file", "--cov-report", "--cov-config", "--deselect",
}
# Options that broaden execution beyond the explicitly named test files.
_PYTEST_BROADENING = {"--doctest-modules"}
# Short flags with no value.
_PYTEST_BOOL_SHORT = set("qvxslEfhV")
# Short flags that consume a value (attached or the next argument).
_PYTEST_VALUE_SHORT = set("kmnrW")
# Short flags that select code/config -> hard deny (except ``-p no:<name>``).
_PYTEST_DENY_SHORT = {"p", "c", "o"}


def resolve_target(value: str, cwd_dir: Path, workspace: Path) -> Path:
    """Resolve a CLI path operand the way the child process would (cwd-relative).

    ``cwd_dir`` and ``workspace`` are already resolved absolute paths and
    ``cwd_dir`` is inside ``workspace``. Both relative (to cwd) and absolute
    operands are accepted *only* when the result stays inside the workspace;
    symlinks are resolved, so a link pointing outside is rejected.
    """
    raw = Path(value)
    resolved = (cwd_dir / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise SecurityError(
            f"路径越界：不允许访问工作区之外的路径（{value}）。"
        ) from exc
    return resolved


def _check_path_value(value: str, cwd_dir: Path, workspace: Path, what: str) -> None:
    candidate = value.split("::", 1)[0]
    # ``--cov-report html:cov_html`` / ``--cov-report=term-missing``
    if ":" in candidate and candidate[1:2] != ":":
        head, _, tail = candidate.partition(":")
        candidate = tail or head
    candidate = candidate.strip()
    if not candidate:
        return
    try:
        resolve_target(candidate, cwd_dir, workspace)
    except SecurityError as exc:
        raise SecurityError(f"{what}（{value}）{exc}") from exc


def _parse_pytest_args(
    argv: list[str], cwd_dir: Path, workspace: Path
) -> tuple[list[str], list[str]]:
    """Walk pytest arguments, validating option semantics instead of prefixes.

    Unknown options are not rejected outright (that would block harmless newer
    flags) but escalated to approval, so an unrecognised flag can never run
    silently. Options known to load code/config are rejected.
    """
    targets: list[str] = []
    unknown: list[str] = []
    positional_only = False
    i = 0
    while i < len(argv):
        token = argv[i]
        if positional_only:
            targets.append(token)
            i += 1
            continue
        if token == "--":
            positional_only = True
            i += 1
            continue
        if token.startswith("--") and len(token) > 2:
            name, sep, value = token.partition("=")
            if name in _PYTEST_HARD_DENY:
                raise SecurityError(
                    f"命令被拒绝：{name} —— {_PYTEST_HARD_DENY[name]}。"
                    "请直接运行测试文件/目录，不要引入插件或外部配置。"
                )
            if name in _PYTEST_VALUE_LONG:
                if not sep:
                    if i + 1 >= len(argv):
                        raise SecurityError(f"命令被拒绝：{name} 缺少参数值。")
                    value = argv[i + 1]
                    i += 1
                if name in _PYTEST_PATH_OPTIONS:
                    _check_path_value(value, cwd_dir, workspace, f"pytest 选项 {name}")
                if name == "--import-mode" and value not in {"prepend", "append", "importlib"}:
                    raise SecurityError(f"命令被拒绝：未知的 --import-mode={value}。")
                i += 1
                continue
            if name in _PYTEST_BOOL_LONG:
                if sep:
                    raise SecurityError(f"命令被拒绝：{name} 不接受参数值（{token}）。")
                i += 1
                continue
            if name in _PYTEST_BROADENING:
                unknown.append(name)
                i += 1
                continue
            unknown.append(name)
            i += 1
            continue
        if token.startswith("-") and token != "-":
            i += 1 + _parse_short_cluster(token, argv, i, cwd_dir, workspace, unknown)
            continue
        targets.append(token)
        i += 1
    return targets, unknown


def _parse_short_cluster(
    token: str, argv: list[str], index: int, cwd_dir: Path, workspace: Path,
    unknown: list[str],
) -> int:
    """Parse a short-option cluster; returns how many extra args were consumed."""
    pos = 1
    while pos < len(token):
        ch = token[pos]
        if ch in _PYTEST_BOOL_SHORT:
            pos += 1
            continue
        if ch in _PYTEST_DENY_SHORT:
            attached = token[pos + 1:]
            consumed = 0
            value = attached
            if not value:
                if index + 1 >= len(argv):
                    raise SecurityError(f"命令被拒绝：-{ch} 缺少参数值。")
                value = argv[index + 1]
                consumed = 1
            if ch == "p":
                # Disabling a plugin (-p no:name) loads nothing; loading one does.
                if not value.startswith("no:"):
                    raise SecurityError(
                        "命令被拒绝：-p/--plugin 会导入并执行插件模块。"
                        "仅允许 -p no:<插件名> 形式用于禁用插件。"
                    )
                return consumed
            raise SecurityError(f"命令被拒绝：-{ch} 会加载外部配置或改写收集规则。")
        if ch in _PYTEST_VALUE_SHORT:
            attached = token[pos + 1:]
            consumed = 0
            value = attached
            if not value:
                if index + 1 >= len(argv):
                    raise SecurityError(f"命令被拒绝：-{ch} 缺少参数值。")
                value = argv[index + 1]
                consumed = 1
            if ch == "n" and not value.isdigit():
                raise SecurityError(f"命令被拒绝：-n 需要正整数（收到 {value}）。")
            return consumed
        # Unknown short flag: escalate rather than guessing its arity.
        unknown.append(f"-{ch}")
        return 0
    return 0


def _validate_test_target(target: str, cwd_dir: Path, workspace: Path) -> None:
    """Validate one explicit pytest target (file, directory or node id)."""
    file_part = target.split("::", 1)[0].strip()
    if not file_part:
        raise SecurityError(f"命令被拒绝：无效的 pytest 目标（{target}）。")
    resolved = resolve_target(file_part, cwd_dir, workspace)
    if resolved.is_dir():
        return
    if resolved.suffix.lower() != ".py":
        raise SecurityError(
            f"命令被拒绝：pytest 只能运行测试文件或目录（{file_part}）。"
        )
    if not _TEST_FILE_RE.match(resolved.name):
        try:
            shown = resolved.relative_to(workspace).as_posix()
        except ValueError:
            shown = file_part
        raise SecurityError(
            f"命令被拒绝：{shown} 不是测试文件（需匹配 test_*.py / *_test.py）。"
            "pytest 会把显式指定的 .py 文件当作模块导入并执行，"
            "因此不能用它运行普通脚本。请改用测试文件，"
            "或用 read_file/edit_file 处理脚本。"
        )


# ---------------------------------------------------------------------------
# Invisible code loading: pytest config files and conftest plugins
# ---------------------------------------------------------------------------

# pytest reads ``addopts`` from pytest.ini / pyproject.toml / tox.ini /
# setup.cfg even when argv names only a test file. Those options never appear
# in the command the user approves, so a run could be re-targeted afterwards by
# editing the config (``addopts = -p evil``). Options that load code/config are
# therefore rejected outright; a config file we cannot parse escalates to
# approval instead of being trusted.
_PYTEST_CONFIG_FILES = ("pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg")
_DENY_IN_ADDOPTS = re.compile(
    r"(?:^|\s)(?:-p|--plugin|-c|--config-file|-o|--override-ini|--pyargs)(?:\s|=|$)"
)
_CONFTEST_PLUGINS = re.compile(r"^\s*pytest_plugins\s*=", re.MULTILINE)


def _pytest_addopts(path: Path) -> str | None:
    """Return the ``addopts`` string of one config file (``None`` = unparsable)."""
    try:
        if path.name == "pyproject.toml":
            import tomllib

            data = tomllib.loads(path.read_text(encoding="utf-8"))
            section = (data.get("tool") or {}).get("pytest") or {}
            options = section.get("ini_options")
            options = section if options is None else options
            value = options.get("addopts", "")
        else:
            import configparser

            parser = configparser.ConfigParser(interpolation=None)
            parser.read(path, encoding="utf-8")
            value = parser.get("pytest", "addopts", fallback="")
    except (OSError, UnicodeDecodeError):
        return None
    except Exception:  # noqa: BLE001 - TOML/INI parsers raise several types
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return None


def _check_pytest_config(cwd_dir: Path, workspace: Path, targets: list[str]) -> str | None:
    """Inspect config/conftest files pytest would load. Returns an approval reason."""
    directories: list[Path] = []
    cursor = cwd_dir
    while True:
        directories.append(cursor)
        if cursor == workspace or cursor.parent == cursor:
            break
        cursor = cursor.parent
    for target in targets:
        resolved = resolve_target(target.split("::", 1)[0], cwd_dir, workspace)
        start = resolved if resolved.is_dir() else resolved.parent
        cursor = start
        while True:
            if cursor not in directories:
                directories.append(cursor)
            if cursor == workspace or cursor.parent == cursor:
                break
            cursor = cursor.parent

    reasons: list[str] = []
    for directory in directories:
        for name in _PYTEST_CONFIG_FILES:
            candidate = directory / name
            if not candidate.is_file():
                continue
            try:
                shown = candidate.relative_to(workspace).as_posix()
            except ValueError:
                shown = str(candidate)
            addopts = _pytest_addopts(candidate)
            if addopts is None:
                reasons.append(f"{shown} 的 pytest 配置无法解析（无法确认 addopts）")
                continue
            match = _DENY_IN_ADDOPTS.search(addopts)
            if match:
                raise SecurityError(
                    f"命令被拒绝：{shown} 的 addopts 含“{match.group(0).strip()}”，"
                    "会在命令行不变的情况下加载插件/外部配置（审批无法覆盖隐藏参数）。"
                    "请先移除该配置项。"
                )
        conftest = directory / "conftest.py"
        if conftest.is_file():
            try:
                text = conftest.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            if _CONFTEST_PLUGINS.search(text):
                try:
                    shown = conftest.relative_to(workspace).as_posix()
                except ValueError:
                    shown = str(conftest)
                reasons.append(f"{shown} 通过 pytest_plugins 加载插件")
    return "；".join(dict.fromkeys(reasons)) or None


def _with_risk(decision: CommandDecision, extra: str | None) -> CommandDecision:
    """Attach an extra risk reason to an already-escalated decision."""
    if not extra:
        return decision
    if not decision.requires_approval:
        return CommandDecision(
            decision.program, decision.effective_program,
            requires_approval=True, risk=extra, notes=decision.notes,
            writes=decision.writes, cache_kind=decision.cache_kind,
        )
    risk = f"{decision.risk}；{extra}" if decision.risk else extra
    return CommandDecision(
        decision.program, decision.effective_program,
        requires_approval=True, risk=risk, notes=decision.notes,
        writes=decision.writes, cache_kind=decision.cache_kind,
    )


def analyze_pytest(argv: list[str], cwd_dir: Path, workspace: Path) -> CommandDecision:
    """Classify a pytest invocation."""
    targets, unknown = _parse_pytest_args(argv, cwd_dir, workspace)
    for target in targets:
        _validate_test_target(target, cwd_dir, workspace)
    config_risk = _check_pytest_config(cwd_dir, workspace, targets)

    if unknown:
        options = "、".join(dict.fromkeys(unknown))
        return _with_risk(
            CommandDecision(
                program="pytest",
                effective_program="pytest",
                requires_approval=True,
                risk=(
                    f"运行 pytest（含未识别/扩大范围的选项 {options}，"
                    "无法静态确认其加载哪些额外代码）"
                ),
                notes=tuple(dict.fromkeys(unknown)),
                cache_kind="pytest",
            ),
            config_risk,
        )

    if not targets:
        return _with_risk(
            CommandDecision(
                program="pytest",
                effective_program="pytest",
                requires_approval=True,
                risk=(
                    "运行 pytest 但未指定测试目标：将递归收集整个项目，"
                    "加载全部 conftest 与配置，执行范围无法静态限定"
                ),
                cache_kind="pytest",
            ),
            config_risk,
        )

    return _with_risk(
        CommandDecision(
            program="pytest",
            effective_program="pytest",
            requires_approval=False,
            notes=tuple(targets),
            cache_kind="pytest",
        ),
        config_risk,
    )


def _validate_ruff(rest: list[str], cwd_dir: Path, workspace: Path) -> CommandDecision:
    """``ruff format`` and ``ruff check --fix`` mutate files -> approval."""
    sub = rest[0].lower() if rest else ""
    mutating = sub == "format" or (
        sub == "check"
        and any(flag in {"--fix", "--fix-only", "--unsafe-fixes"} for flag in rest[1:])
    )
    for token in rest[1:]:
        if token.startswith(("--config", "-c")):
            raise SecurityError("命令被拒绝：ruff 不接受外部配置文件参数。")
        if not token.startswith("-"):
            _check_path_value(token, cwd_dir, workspace, "ruff 目标")
    if mutating:
        targets = [t for t in rest[1:] if not t.startswith("-")]
        writes: list[str] = []
        for token in targets or ["."]:
            target = resolve_target(token, cwd_dir, workspace)
            rel = _rel_posix(target, workspace)
            writes.append(rel + "/" if target.is_dir() else rel)
        return CommandDecision(
            program="ruff", effective_program="ruff", requires_approval=True,
            risk="执行会修改项目文件的命令（格式化/自动修复）",
            writes=tuple(writes),
        )
    return CommandDecision(program="ruff", effective_program="ruff")


def _validate_mypy(rest: list[str], cwd_dir: Path, workspace: Path) -> CommandDecision:
    """``mypy`` is read-only, but path arguments must stay in the workspace."""
    for token in rest:
        if token.startswith("--config-file"):
            raise SecurityError(
                "命令被拒绝：mypy 的 --config-file 可从任意位置加载配置（可引入插件）。"
            )
        if token in {"-p", "--package"} or token.startswith("--package="):
            raise SecurityError("命令被拒绝：mypy 的 --package 会按模块名导入代码。")
        if token.startswith("-"):
            continue
        if "/" in token or "\\" in token or token.endswith(".py"):
            _check_path_value(token, cwd_dir, workspace, "mypy 目标指向工作区外")
    return CommandDecision(program="mypy", effective_program="mypy", cache_kind="mypy")


# Tokens in a ``package.json`` script body that compose commands or invoke an
# interpreter/downloader. The body never appears in argv, exactly like pytest's
# addopts, so such a script is escalated instead of trusted.
_NPM_SCRIPT_RISK = re.compile(
    r"(?:&&|\|\||;|\||>|<|\$\(|`|\bnode\s+-e\b|\bpython3?\s+-c\b|\bcurl\b|\bwget\b)",
    re.IGNORECASE,
)


def _npm_script_risk(cwd_dir: Path, workspace: Path, script: str) -> str | None:
    """Approval reason when a package.json script cannot be statically bounded."""
    directory = cwd_dir
    while True:
        candidate = directory / "package.json"
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError):
                return "package.json 无法解析，无法确认 npm 脚本内容"
            scripts = data.get("scripts") if isinstance(data, dict) else None
            body = scripts.get(script) if isinstance(scripts, dict) else None
            if not isinstance(body, str):
                return None
            match = _NPM_SCRIPT_RISK.search(body)
            if match:
                try:
                    shown = candidate.relative_to(workspace).as_posix()
                except ValueError:
                    shown = str(candidate)
                return (
                    f"{shown} 的脚本“{script}”含“{match.group(0)}”"
                    f"（{body.strip()[:100]}），可执行任意命令，无法静态限定"
                )
            return None
        if directory == workspace or directory.parent == directory:
            return None
        directory = directory.parent


def _validate_coverage(rest: list[str], cwd_dir: Path, workspace: Path) -> CommandDecision:
    """``coverage run <script>`` is a generic code executor -> only pytest."""
    if not rest or rest[0].lower() != "run":
        return CommandDecision(
            program="coverage", effective_program="coverage", requires_approval=True,
            risk="执行 coverage 子命令（会读取/写入覆盖率数据文件）",
            cache_kind="coverage",
        )
    args = rest[1:]
    if "-m" not in args:
        raise SecurityError(
            "命令被拒绝：coverage run <脚本> 会执行任意 Python 文件。"
            "仅允许 coverage run -m pytest <测试目标> 形式。"
        )
    idx = args.index("-m")
    if idx + 1 >= len(args):
        raise SecurityError("命令被拒绝：coverage run -m 缺少模块名。")
    module = args[idx + 1]
    inner = args[idx + 2:]
    if module != "pytest":
        raise SecurityError(
            f"命令被拒绝：coverage run -m {module} 会执行非测试模块。"
            "仅允许 coverage run -m pytest。"
        )
    decision = analyze_pytest(inner, cwd_dir, workspace)
    return CommandDecision(
        program="coverage", effective_program="pytest",
        requires_approval=decision.requires_approval, risk=decision.risk,
        notes=decision.notes, cache_kind="coverage",
    )


def _validate_unittest(rest: list[str], cwd_dir: Path, workspace: Path) -> CommandDecision:
    """``unittest`` accepts module names that are imported and executed."""
    if rest and rest[0].lower() == "discover":
        for i, token in enumerate(rest):
            if token in {"-s", "--start-directory", "-t", "--top-level-directory"}:
                if i + 1 < len(rest):
                    _check_path_value(
                        rest[i + 1], cwd_dir, workspace, "unittest 目录指向工作区外"
                    )
            elif token.startswith("-s") and len(token) > 2:
                _check_path_value(
                    token[2:], cwd_dir, workspace, "unittest 目录指向工作区外"
                )
        return CommandDecision(
            program="unittest", effective_program="unittest", requires_approval=True,
            risk="unittest discover 会递归导入测试模块（执行范围无法静态限定）",
            cache_kind="unittest",
        )
    if not rest:
        return CommandDecision(
            program="unittest", effective_program="unittest", requires_approval=True,
            risk="unittest 会按模块名导入并执行代码",
            cache_kind="unittest",
        )
    raise SecurityError(
        "命令被拒绝：python -m unittest <模块名> 会按名称导入并执行任意模块；"
        "请改用 pytest 指定测试文件。"
    )


def _validate_compileall(
    rest: list[str], cwd_dir: Path, workspace: Path
) -> CommandDecision:
    """``compileall`` writes ``.pyc`` files and accepts arbitrary paths."""
    for token in rest:
        if token.startswith("-"):
            continue
        _check_path_value(token, cwd_dir, workspace, "compileall 目标指向工作区外")
    return CommandDecision(
        program="compileall", effective_program="compileall", requires_approval=True,
        risk="compileall 会写入 .pyc 文件（生成字节码缓存）",
        cache_kind="compileall",
    )


# ---------------------------------------------------------------------------
# Interpreter shadowing
# ---------------------------------------------------------------------------


def _interpreter_owned(path: Path) -> bool:
    """True when ``path`` belongs to the active interpreter's own environment.

    A project venv often lives *inside* the workspace, so ``pytest``/``ruff``
    legitimately resolve under the workspace. Those are not shadowing.
    """
    roots = {
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        Path(sys.executable).resolve().parent.parent,
    }
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def check_module_shadowing(module: str, workspace: Path) -> None:
    """Reject ``python -m X`` when the workspace can shadow module ``X``.

    ``python -m`` puts the working directory first on ``sys.path``, so a file
    named ``pytest.py`` inside the project would be imported *instead of* the
    real test runner — arbitrary code execution with a fully valid command
    line. This is checkable before spawning the child.
    """
    short = module.split(".")[0]
    for candidate in (workspace / f"{short}.py", workspace / short / "__init__.py"):
        if candidate.exists():
            raise SecurityError(
                f"命令被拒绝：工作区内存在 {candidate.name}，会遮蔽真实模块“{short}”。"
                "python -m 会把工作目录放在 sys.path 首位，属于模块劫持，"
                "请删除/重命名该文件后再运行。"
            )
    try:
        spec = importlib.util.find_spec(short)
    except (ImportError, ValueError, AttributeError):
        return
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin:
        return
    try:
        resolved = Path(origin).resolve()
    except OSError:
        return
    try:
        resolved.relative_to(workspace)
    except ValueError:
        return
    if _interpreter_owned(resolved):
        return
    raise SecurityError(
        f"命令被拒绝：模块“{short}”解析到工作区内的 {resolved.name}，疑似模块劫持。"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def classify(
    program: str,
    args: list[str],
    cwd: str,
    workspace: str | Path | None = None,
) -> CommandDecision:
    """Classify a parameterized command. Raises ``SecurityError`` to reject.

    ``workspace`` is the run's resolved workspace root (falls back to
    ``config.WORKSPACE_PATH``); ``cwd`` is resolved against it and must itself
    stay inside it.
    """
    from app.tools.security import resolve_workspace_path

    program = (program or "").strip().lower()
    argv = [str(a) for a in args]
    if not program:
        raise SecurityError("命令为空：缺少可执行程序名称。")
    if program != "pip" and program not in _ALLOWED_PROGRAMS:
        raise SecurityError(f"不允许的命令：程序“{program}”不在白名单中。")

    cwd_dir = resolve_workspace_path(cwd or ".", workspace)
    ws = Path(workspace).resolve() if workspace is not None else cwd_dir

    joined = " ".join([program, *argv])
    for pattern in DENY_ARG_PATTERNS:
        if re.search(pattern, joined, flags=re.IGNORECASE):
            raise SecurityError(f"命令被拒绝：包含危险模式（{pattern}）。")

    if program == "pip":
        return CommandDecision(
            program, "pip", requires_approval=True, risk="安装或修改项目依赖",
            writes=dependency_writes("python", cwd_dir, ws),
        )
    if program == "pytest":
        return analyze_pytest(argv, cwd_dir, ws)
    if program == "npm":
        if argv and argv[0].lower() == "test":
            risk = _npm_script_risk(cwd_dir, ws, "test")
            return CommandDecision(
                program, "npm", requires_approval=bool(risk), risk=risk,
                cache_kind="npm",
            )
        if argv and argv[0].lower() == "run":
            script = argv[1].lower() if len(argv) > 1 else ""
            if script in _VERIFYING_NPM_SCRIPTS:
                risk = _npm_script_risk(cwd_dir, ws, script)
                return CommandDecision(
                    program, "npm", requires_approval=bool(risk), risk=risk,
                    cache_kind="npm",
                )
            return CommandDecision(
                program, "npm", requires_approval=True,
                risk=f"执行 npm 脚本“{script or '?'}”（package.json 脚本可执行任意命令）",
            )
        if argv and argv[0].lower() in _APPROVAL_SUBCOMMANDS.get(program, set()):
            return CommandDecision(
                program, "npm", requires_approval=True, risk="安装或修改项目依赖",
                writes=dependency_writes("npm", cwd_dir, ws),
            )
        raise SecurityError(f"命令被拒绝：“npm {argv[0] if argv else ''}”不在允许列表中。")
    if program == "npx":
        if not argv or argv[0].lower() != "tsc":
            raise SecurityError("命令被拒绝：npx 仅允许运行 tsc。")
        return CommandDecision(program, "npx", cache_kind="tsc")
    if program in {"python", "uv"}:
        if not argv:
            raise SecurityError(f"命令被拒绝：“{program}”需要子命令参数。")
        sub = argv[0]
        if sub in _APPROVAL_SUBCOMMANDS.get(program, set()):
            return CommandDecision(
                program, program, requires_approval=True, risk="安装或修改项目依赖",
                writes=dependency_writes("python", cwd_dir, ws),
            )
        if sub not in _ALLOWED_PROGRAMS[program]:
            raise SecurityError(f"命令被拒绝：“{program} {sub}”不在允许列表中。")
        if sub == "--version":
            return CommandDecision(program, program)
        if sub == "-m":
            if len(argv) < 2:
                raise SecurityError("命令被拒绝：“python -m”需要模块名。")
            module = argv[1]
            rest = argv[2:]
            if module not in _KNOWN_MODULES:
                raise SecurityError(f"命令被拒绝：不允许运行模块“{module}”。")
            check_module_shadowing(module, ws)
            if module == "pytest":
                return analyze_pytest(rest, cwd_dir, ws)
            if module == "ruff":
                return _validate_ruff(rest, cwd_dir, ws)
            if module == "mypy":
                return _validate_mypy(rest, cwd_dir, ws)
            if module == "coverage":
                return _validate_coverage(rest, cwd_dir, ws)
            if module == "unittest":
                return _validate_unittest(rest, cwd_dir, ws)
            if module == "compileall":
                return _validate_compileall(rest, cwd_dir, ws)
            return CommandDecision(  # pip
                program, "pip", requires_approval=True, risk="安装或修改项目依赖",
                writes=dependency_writes("python", cwd_dir, ws),
            )
        if sub == "run":  # uv run <prog>
            if len(argv) < 2:
                raise SecurityError("命令被拒绝：“uv run”需要目标程序。")
            target = argv[1]
            rest = argv[2:]
            if target not in _KNOWN_MODULES:
                raise SecurityError(f"命令被拒绝：不允许通过 uv run 运行“{target}”。")
            check_module_shadowing(target, ws)
            if target == "pytest":
                return analyze_pytest(rest, cwd_dir, ws)
            if target == "ruff":
                return _validate_ruff(rest, cwd_dir, ws)
            if target == "mypy":
                return _validate_mypy(rest, cwd_dir, ws)
            if target == "pip":
                return CommandDecision(
                    program, "pip", requires_approval=True, risk="安装或修改项目依赖",
                    writes=dependency_writes("python", cwd_dir, ws),
                )
            raise SecurityError(f"命令被拒绝：不允许通过 uv run 运行“{target}”。")
        raise SecurityError(f"命令被拒绝：“{program} {sub}”不在允许列表中。")
    if program == "ruff":
        return _validate_ruff(argv, cwd_dir, ws)
    return _validate_mypy(argv, cwd_dir, ws)  # mypy




def inject_cache_argv(decision: CommandDecision, argv: list[str], scratch_dir: str) -> list[str]:
    """Append CLI flags that pin caches into ``scratch_dir`` (idempotent)."""
    kind = decision.cache_kind
    if not kind:
        return list(argv)
    out = list(argv)
    joined = " ".join(out)
    wants_pytest_cache = (
        kind == "pytest"
        or (kind in {"coverage", "unittest"} and decision.effective_program == "pytest")
    )
    if wants_pytest_cache and "cache_dir=" not in joined and "--cache-dir" not in joined:
        out.extend(["-o", f"cache_dir={scratch_dir}/pytest_cache"])
    if kind == "mypy" and "--cache-dir" not in joined:
        out.append(f"--cache-dir={scratch_dir}/mypy_cache")
    if kind == "tsc" and "--tsBuildInfoFile" not in joined:
        # Prefer writing incremental info into scratch (env covers npm caches).
        out.append(f"--tsBuildInfoFile={scratch_dir}/tsbuildinfo/tsconfig.tsbuildinfo")
    return out


# ---------------------------------------------------------------------------
# Environment sanitation
# ---------------------------------------------------------------------------


def sanitize_inherited_env(env: dict[str, str]) -> list[str]:
    """Drop inherited vars that can redirect which code an interpreter loads."""
    dropped = [key for key in list(env) if key.casefold() in BLOCKED_ENV_KEYS]
    for key in dropped:
        env.pop(key, None)
    return dropped


def filter_model_env(env: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Drop model-supplied env vars that could hijack resolution or imports."""
    kept: dict[str, str] = {}
    dropped: list[str] = []
    for key, value in env.items():
        if str(key).casefold() in MODEL_BLOCKED_ENV_KEYS:
            dropped.append(str(key))
        else:
            kept[key] = value
    return kept, dropped


def python_interpreter_flags(program: str, argv: list[str]) -> list[str]:
    """Extra flags for ``python -m`` runs (deterministic, no env surprises).

    ``-E`` ignores ``PYTHON*`` env vars (the child env is already sanitised, so
    this is defense in depth) and ``-B`` avoids writing ``__pycache__`` for
    verification runs. We intentionally do NOT pass ``-P``: pytest relies on
    the project root being importable, and shadowing is handled by
    :func:`check_module_shadowing` instead.
    """
    if program == "python" and argv and argv[0] == "-m":
        return ["-E", "-B"]
    return []
