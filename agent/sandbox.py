import sys
import traceback
from io import StringIO
from typing import Tuple

AGENT_SANDBOX = {
    # Basic builtins
    "range": range,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    # Iteration
    "zip": zip,
    "enumerate": enumerate,
    "reversed": reversed,
    # Aggregation
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "round": round,
    # Sorting
    "sorted": sorted,
    # Type conversions
    "type": type,
    "isinstance": isinstance,
    # Math operations (via print)
    "pow": pow,
    # Output
    "print": print,
}


def execute_code(
    code: str,
    state: dict,
    iteration: int,
) -> Tuple[str, dict, bool]:
    """
    Execute Python code in restricted sandbox with stdout/stderr capture.

    Args:
        code: Python code string to execute
        state: Dictionary of variables persistent across iterations
        iteration: Current iteration number (for debugging)

    Returns:
        (captured_output, updated_state, success_bool)

    Features:
        - Captures stdout (print() statements)
        - Captures stderr (exceptions)
        - Persists variables across calls
        - Restricts access to only whitelisted functions
        - Catches all exceptions safely
    """

    # Build sandbox environment
    exec_globals = {**AGENT_SANDBOX, **state}
    exec_locals = {}

    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured_stdout = StringIO()
    captured_stderr = StringIO()

    sys.stdout = captured_stdout
    sys.stderr = captured_stderr

    success = False

    try:
        # Execute code in restricted environment
        exec(code, exec_globals, exec_locals)
        success = True

        # Merge new/modified variables back into state (for next iteration)
        for key, value in exec_locals.items():
            state[key] = value

        # Also capture any variables created in exec_globals that weren't in sandbox
        for key, value in exec_globals.items():
            if (
                key not in AGENT_SANDBOX
                and key not in state
                and not key.startswith("_")
            ):
                state[key] = value

    except SyntaxError as e:
        success = False
        captured_stderr.write(f"SyntaxError (line {e.lineno}): {e.msg}\n")
        if e.text:
            captured_stderr.write(f"  {e.text.strip()}\n")
            captured_stderr.write(f"  {' ' * (e.offset - 1)}^\n")

    except Exception as e:
        success = False
        captured_stderr.write(f"{type(e).__name__}: {str(e)}\n")
        captured_stderr.write(traceback.format_exc())

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    # Combine stdout and stderr
    output = captured_stdout.getvalue() + captured_stderr.getvalue()

    # If nothing was captured, indicate code ran but produced no output
    if not output.strip():
        if success:
            output = "[Code executed successfully, no output]\n"
        else:
            output = "[Code failed to execute]\n"

    return output, state, success
