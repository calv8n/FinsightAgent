#!/usr/bin/env python3
"""
sandbox/runner.py

Secure Docker sandbox runner for executing agent-generated Python code.

Protocol:
    stdin:
        {
            "code": "...",
            "state": {...}
        }

    stdout:
        {
            "stdout": "...",
            "stderr": "...",
            "state": {...},
            "success": bool
        }

Security Layers:
    1. Restricted builtins
    2. No unrestricted imports
    3. Audit hook blocks dangerous imports/open
    4. Docker isolation
    5. Resource limits
    6. Non-root user
    7. No network
"""

import json
import resource
import sys
import traceback
from io import StringIO

# =============================================================================
# IMPORT SAFE LIBRARIES EARLY
# =============================================================================

import numpy as np
import pandas as pd

# =============================================================================
# RESOURCE LIMITS
# =============================================================================


def _apply_resource_limits() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    except Exception:
        pass


# =============================================================================
# SAFE IMPORT SYSTEM
# =============================================================================

_ALLOWED_IMPORTS = {
    "numpy",
    "pandas",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]

    if root in _ALLOWED_IMPORTS:
        return __import__(name, globals, locals, fromlist, level)

    raise PermissionError(f"[sandbox] import blocked: {name}")


# =============================================================================
# SAFE BUILTINS
# =============================================================================

_SAFE_BUILTINS = {
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
    "__import__": _safe_import,
}

# =============================================================================
# BLOCKED MODULES
# =============================================================================

_BLOCKED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "glob",
    "tempfile",
    "requests",
    "httpx",
    "urllib",
    "ctypes",
    "multiprocessing",
    "signal",
    "pickle",
    "marshal",
    "builtins",
    "resource",
}


def _patch_sys_modules() -> None:
    for mod in _BLOCKED_MODULES:
        if mod in sys.modules:
            del sys.modules[mod]


# =============================================================================
# AUDIT HOOK
# =============================================================================

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
        name = args[0] if args else ""
        root = name.split(".")[0]

        if root in _ALLOWED_IMPORT_ROOTS:
            return

        if name.startswith("_"):
            return

        raise PermissionError(f"[sandbox] import blocked: {name!r}")

    elif event == "open":
        path = str(args[0]) if args else ""

        allowed_paths = (
            "/usr/local/lib/python",
            "/usr/lib/python",
            "<agent>",
            "",
        )

        if any(path.startswith(p) for p in allowed_paths):
            return

        raise PermissionError(f"[sandbox] open blocked: {path!r}")


# =============================================================================
# STATE SERIALIZATION
# =============================================================================


def _safe_serialise(state: dict) -> dict:
    out = {}

    for k, v in state.items():
        if k.startswith("_"):
            continue

        try:
            json.dumps(v)
            out[k] = v

        except (TypeError, ValueError):

            # NumPy arrays
            if isinstance(v, np.ndarray):
                out[k] = {
                    "__type__": "numpy.ndarray",
                    "dtype": str(v.dtype),
                    "shape": v.shape,
                    "value": v.tolist(),
                }

            # Pandas DataFrame
            elif isinstance(v, pd.DataFrame):
                out[k] = {
                    "__type__": "pandas.dataframe",
                    "value": v.to_dict(),
                }

            # Pandas Series
            elif isinstance(v, pd.Series):
                out[k] = {
                    "__type__": "pandas.series",
                    "value": v.to_dict(),
                }

            else:
                out[k] = repr(v)

    return out


def _safe_deserialise(raw: dict) -> dict:
    out = {}

    for k, v in raw.items():

        if not isinstance(k, str):
            continue

        if isinstance(v, dict):

            obj_type = v.get("__type__")

            # Restore numpy arrays
            if obj_type == "numpy.ndarray":
                out[k] = np.array(v["value"], dtype=v.get("dtype"))

            # Restore pandas dataframe
            elif obj_type == "pandas.dataframe":
                out[k] = pd.DataFrame(v["value"])

            # Restore pandas series
            elif obj_type == "pandas.series":
                out[k] = pd.Series(v["value"])

            else:
                out[k] = v

        else:
            out[k] = v

    return out


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def run(code: str, state: dict) -> dict:
    _apply_resource_limits()

    _patch_sys_modules()

    sys.addaudithook(_audit_hook)

    exec_env = {
        "__builtins__": _SAFE_BUILTINS,
        "np": np,
        "pd": pd,
        **state,
    }

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    cap_out = StringIO()
    cap_err = StringIO()

    sys.stdout = cap_out
    sys.stderr = cap_err

    success = False

    try:
        exec(compile(code, "<agent>", "exec"), exec_env)

        success = True

        # Persist state
        reserved = {
            "__builtins__",
            "np",
            "pd",
        }

        for k, v in exec_env.items():
            if not k.startswith("_") and k not in reserved:
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
        stdout_val = "[no output]\n" if success else "[execution failed]\n"

    return {
        "stdout": stdout_val,
        "stderr": stderr_val,
        "state": _safe_serialise(state),
        "success": success,
    }


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    try:
        payload = json.loads(sys.stdin.read())

        code = payload.get("code", "")
        raw_state = payload.get("state", {})

        state = _safe_deserialise(raw_state)

        result = run(code, state)

        sys.stdout.write(json.dumps(result))
        sys.stdout.flush()

    except Exception as exc:
        error = {
            "stdout": "",
            "stderr": f"Fatal sandbox error: {type(exc).__name__}: {exc}",
            "state": {},
            "success": False,
        }

        sys.stdout.write(json.dumps(error))
        sys.stdout.flush()
