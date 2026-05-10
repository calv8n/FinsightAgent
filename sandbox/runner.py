#!/usr/bin/env python3
"""
sandbox/runner.py — Executes inside the Docker container.

Protocol:
  stdin:  JSON  {"code": "...", "state": {...}}
  stdout: JSON  {"stdout": "...", "stderr": "...", "state": {...}, "success": bool}

Security layers:
  1. __builtins__ replaced with a safe whitelist — no open/eval/exec/__import__
  2. sys.modules patched to block dangerous modules
  3. Audit hook blocks stray import/open attempts from agent code
  4. Resource limits (CPU + memory)
  5. Runs as non-root UID 10001 (Dockerfile)
  6. --network none, --cap-drop ALL (docker run flags on host)

numpy/pandas are NOT imported at module level.
They are imported lazily inside run() so the audit hook can be installed first,
and the import happens inside the exec environment where it is fully controlled.
"""

import json
import resource
import sys
import traceback
from io import StringIO

# ============================================================================
# RESOURCE LIMITS
# ============================================================================


def _apply_resource_limits() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    except Exception:
        pass  # not all kernels support this; Docker's --memory is the real cap


# ============================================================================
# SAFE BUILTINS WHITELIST
# ============================================================================

_SAFE_BUILTINS: dict = {
    "None": None,
    "True": True,
    "False": False,
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "str": str,
    "bytes": bytes,
    "bytearray": bytearray,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "iter": iter,
    "next": next,
    "map": map,
    "filter": filter,
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "pow": pow,
    "divmod": divmod,
    "sorted": sorted,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "id": id,
    "hash": hash,
    "repr": repr,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "oct": oct,
    "bin": bin,
    "format": format,
    "any": any,
    "all": all,
    "print": print,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "ZeroDivisionError": ZeroDivisionError,
    "StopIteration": StopIteration,
    "RuntimeError": RuntimeError,
    "NotImplementedError": NotImplementedError,
}


# ============================================================================
# MODULE BLOCKLIST — set to None so import returns None immediately
# Does NOT include threading/_thread: Python's atexit needs them at shutdown
# ============================================================================

_BLOCKED_MODULES: set = {
    "os",
    "os.path",
    "pathlib",
    "shutil",
    "glob",
    "tempfile",
    "fileinput",
    "fnmatch",
    "stat",
    "socket",
    "ssl",
    "http",
    "http.client",
    "http.server",
    "urllib",
    "urllib.request",
    "urllib.parse",
    "urllib.error",
    "requests",
    "httpx",
    "aiohttp",
    "ftplib",
    "smtplib",
    "imaplib",
    "poplib",
    "telnetlib",
    "xmlrpc",
    "subprocess",
    "multiprocessing",
    "signal",
    "ctypes",
    "cffi",
    "mmap",
    "platform",
    "sysconfig",
    "site",
    "pickle",
    "pickletools",
    "shelve",
    "marshal",
    "ast",
    "dis",
    "py_compile",
    "compileall",
    "pkgutil",
    "pdb",
    "trace",
    "pty",
    "tty",
    "termios",
    "curses",
    "resource",
    "builtins",
}


def _patch_sys_modules() -> None:
    for mod in _BLOCKED_MODULES:
        sys.modules[mod] = None  # type: ignore[assignment]


# ============================================================================
# AUDIT HOOK
# Blocks import/open events from agent exec'd code.
# Installed AFTER numpy/pandas finish loading so their internals don't trip it.
# Does NOT block "exec" or "compile" — runner.py calls both itself.
# ============================================================================

_ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "numpy",
        "pandas",
        "pytz",
        "dateutil",
        "six",
    }
)


def _audit_hook(event: str, args: tuple) -> None:
    if event == "import":
        name: str = args[0] if args else ""
        # Allow numpy, pandas and all their sub-packages
        root = name.split(".")[0]
        if root in _ALLOWED_IMPORT_ROOTS:
            return
        # Allow private/internal names Python uses during import machinery
        if name.startswith("_"):
            return
        raise PermissionError(f"[sandbox] import blocked: {name!r}")

    elif event == "open":
        path: str = str(args[0]) if args else ""
        _ok = (
            "/usr/local/lib/python",
            "/usr/lib/python",
            "/usr/local/lib/",
            "/usr/lib/",
            "<agent>",
            "",
        )
        if any(path.startswith(p) for p in _ok):
            return
        raise PermissionError(f"[sandbox] open blocked: {path!r}")


# ============================================================================
# STATE SERIALISATION
# ============================================================================


def _safe_deserialise(raw: dict) -> dict:
    allowed = (type(None), bool, int, float, str, list, dict)
    return {
        k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, allowed)
    }


def _safe_serialise(state: dict) -> dict:
    out: dict = {}
    for k, v in state.items():
        if k.startswith("_"):
            continue
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = repr(v)
    return out


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def run(code: str, state: dict) -> dict:
    _apply_resource_limits()

    # Import numpy and pandas HERE (not at module level) so that:
    # a) they are loaded before the audit hook is installed
    # b) any ImportError (e.g. missing package) is caught and returned cleanly
    try:
        import numpy as np  # noqa: F401  (used in safe_globals below)
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        return {
            "stdout": "",
            "stderr": f"ImportError loading numpy/pandas: {exc}\n",
            "state": _safe_serialise(state),
            "success": False,
        }

    # Block dangerous modules, then install the audit hook.
    # numpy/pandas are already in sys.modules, so their lazy sub-imports
    # that happen during exec() are already cached and won't re-trigger.
    _patch_sys_modules()
    sys.addaudithook(_audit_hook)

    safe_globals: dict = {
        "__builtins__": _SAFE_BUILTINS,
        "np": np,
        "pd": pd,
        **state,
    }
    exec_locals: dict = {}

    old_stdout, old_stderr = sys.stdout, sys.stderr
    cap_out, cap_err = StringIO(), StringIO()
    sys.stdout = cap_out
    sys.stderr = cap_err
    success = False

    try:
        exec(compile(code, "<agent>", "exec"), safe_globals, exec_locals)
        success = True

        for k, v in exec_locals.items():
            if not k.startswith("_"):
                state[k] = v

        _reserved = {"np", "pd", "__builtins__"} | set(_SAFE_BUILTINS)
        for k, v in safe_globals.items():
            if not k.startswith("_") and k not in _reserved and k not in state:
                state[k] = v

    except SyntaxError as exc:
        cap_err.write(f"SyntaxError (line {exc.lineno}): {exc.msg}\n")
        if exc.text:
            cap_err.write(f"  {exc.text.rstrip()}\n")
            cap_err.write(f"  {' ' * max(0, (exc.offset or 1) - 1)}^\n")

    except PermissionError as exc:
        cap_err.write(f"SecurityError: {exc}\n")

    except Exception as exc:
        cap_err.write(f"{type(exc).__name__}: {exc}\n")
        cap_err.write(traceback.format_exc())

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    stdout_val = cap_out.getvalue()
    stderr_val = cap_err.getvalue()

    if not stdout_val.strip() and not stderr_val.strip():
        stdout_val = "[no output]\n" if success else "[execution failed, no output]\n"

    return {
        "stdout": stdout_val,
        "stderr": stderr_val,
        "state": _safe_serialise(state),
        "success": success,
    }


if __name__ == "__main__":
    payload = json.loads(sys.stdin.read())
    code = payload.get("code", "")
    raw_state = payload.get("state", {})
    state = _safe_deserialise(raw_state)
    result = run(code, state)
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()
