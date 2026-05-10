"""
agent/sandbox_client.py

Host-side client that runs code inside the hardened Docker sandbox.

Each call to SandboxClient.run() does:
  1. docker run  (new container per call — clean slate security-wise)
  2. Write JSON payload to container stdin
  3. Read JSON result from container stdout
  4. Kill container after timeout
  5. Return structured result

The container has:
  - --network none              (no outbound or inbound network)
  - --tmpfs /tmp (noexec)       (only writable path; wiped on exit)
  - --tmpfs /run (noexec)       (needed by some libs at startup)
  - --memory 512m               (OOM kill if exceeded)
  - --cpus 0.5                  (CPU throttle)
  - --cap-drop ALL              (no Linux capabilities)
  - --security-opt no-new-privileges
  - non-root UID 10001          (cannot write to /app or install packages)
  Note: --read-only is NOT used — it prevents numpy/pandas .so files from
  being mmap'd with PROT_EXEC. Write protection comes from non-root + cap-drop.
"""

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any

# ============================================================================
# CONFIG
# ============================================================================

DOCKER_IMAGE = "finsight-sandbox"
EXEC_TIMEOUT = 30  # numpy/pandas warm-start needs ~2-3s
MEMORY_LIMIT = "512m"  # 128m OOMs OpenBLAS on import
MEMORY_SWAP = "512m"
CPU_LIMIT = "0.5"
TMPFS_SIZE = "16m"

# OpenBLAS spins up a thread pool on import; capping at 1 thread avoids
# the multi-threaded memory allocation that OOMs inside a constrained container.
SANDBOX_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


# ============================================================================
# RESULT TYPE
# ============================================================================


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    state: dict[str, Any]  # updated state to pass to next iteration
    ok: bool  # True = code ran without exception
    elapsed: float  # wall-clock seconds

    @property
    def output(self) -> str:
        """Convenience: combined stdout + stderr for feeding to agent."""
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout)
        if self.stderr.strip():
            parts.append(f"[stderr]\n{self.stderr}")
        return "\n".join(parts) if parts else "[no output]"


# ============================================================================
# SANDBOX CLIENT
# ============================================================================


class SandboxClient:
    """
    Runs Python code in a locked-down Docker container.

    Usage:
        client = SandboxClient()
        result = client.run("x = 1 + 1\nprint(x)", state={})
        print(result.stdout)   # "2"
        print(result.state)    # {"x": 2}
    """

    def __init__(
        self,
        image: str = DOCKER_IMAGE,
        timeout: int = EXEC_TIMEOUT,
    ):
        self.image = image
        self.timeout = timeout
        self._verify_image()

    # ------------------------------------------------------------------ #
    def _verify_image(self):
        """Check the Docker image exists. Raise if not."""
        result = subprocess.run(
            ["docker", "image", "inspect", self.image],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Docker image '{self.image}' not found.\n"
                f"Build it first:\n"
                f"  cd sandbox && docker build -t {self.image} .\n"
            )

    # ------------------------------------------------------------------ #
    def run(self, code: str, state: dict[str, Any]) -> ExecResult:
        """
        Execute code inside the sandbox container.

        Args:
            code:  Python source code string
            state: Variables from previous iterations (JSON-serialisable)

        Returns:
            ExecResult with stdout, stderr, updated state, ok flag, elapsed time
        """
        payload = json.dumps({"code": code, "state": state})

        # Build env-var flags for OPENBLAS / OMP thread caps
        env_flags: list[str] = []
        for k, v in SANDBOX_ENV.items():
            env_flags += ["--env", f"{k}={v}"]

        cmd = [
            "docker",
            "run",
            "--rm",
            "--interactive",
            "--network",
            "none",  # zero network access
            # --read-only REMOVED: it prevents numpy/pandas .so files from being
            # mmap'd with PROT_EXEC, causing "failed to map segment from shared object".
            # Filesystem writes are still blocked by running as non-root (UID 10001)
            # with no write permission on /app, and by --cap-drop ALL.
            "--tmpfs",
            "/tmp:size=64m,noexec,nosuid",  # only writable dir
            "--tmpfs",
            "/run:size=8m,noexec,nosuid",  # needed by some libs at startup
            "--memory",
            MEMORY_LIMIT,
            "--memory-swap",
            MEMORY_SWAP,
            "--cpus",
            CPU_LIMIT,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            *env_flags,
            self.image,
        ]

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = time.monotonic() - t0

            if proc.returncode != 0:
                # Docker-level failure (OOM kill, image error, etc.)
                return ExecResult(
                    stdout="",
                    stderr=f"Container error (exit {proc.returncode}): {proc.stderr.strip()}",
                    state=state,
                    ok=False,
                    elapsed=elapsed,
                )

            # Parse JSON result from runner.py
            try:
                result = json.loads(proc.stdout)
                return ExecResult(
                    stdout=result.get("stdout", ""),
                    stderr=result.get("stderr", ""),
                    state=result.get("state", state),
                    ok=result.get("success", False),  # runner.py uses "success"
                    elapsed=elapsed,
                )
            except json.JSONDecodeError:
                return ExecResult(
                    stdout="",
                    stderr=f"Runner returned non-JSON: {proc.stdout[:200]}",
                    state=state,
                    ok=False,
                    elapsed=elapsed,
                )

        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            return ExecResult(
                stdout="",
                stderr=f"Execution timed out after {self.timeout}s",
                state=state,
                ok=False,
                elapsed=elapsed,
            )


# ============================================================================
# IMAGE BUILD HELPER
# ============================================================================


def build_sandbox_image(image: str = DOCKER_IMAGE) -> bool:
    """
    Build the Docker sandbox image from sandbox/Dockerfile.
    Call once before first use, or after changing the Dockerfile/runner.py.

    Returns True on success, False on failure.
    """
    import os

    # sandbox/ sits next to the agent/ directory
    sandbox_dir = os.path.join(os.path.dirname(__file__), "..", "sandbox")
    sandbox_dir = os.path.abspath(sandbox_dir)

    if not os.path.isfile(os.path.join(sandbox_dir, "Dockerfile")):
        print(f"❌ Dockerfile not found in {sandbox_dir}")
        return False

    print(f"Building {image!r} from {sandbox_dir} ...")
    result = subprocess.run(
        ["docker", "build", "-t", image, sandbox_dir],
        # stream build output directly to the terminal
        capture_output=False,
    )
    if result.returncode == 0:
        print(f"✅ Image built: {image}")
        return True
    else:
        print(f"❌ Build failed (exit {result.returncode})")
        return False


# ============================================================================
# SMOKE TEST (run directly: python sandbox_client.py)
# ============================================================================

if __name__ == "__main__":
    client = SandboxClient()

    tests = [
        # (label, code, initial_state, expected_in_stdout)
        (
            "Basic arithmetic",
            "x = 6 * 7\nprint(x)",
            {},
            "42",
        ),
        (
            "State persistence (pass state in)",
            "print(x + 10)",
            {"x": 42},
            "52",
        ),
        (
            "Blocked import: os",
            "import os\nprint(os.listdir('/'))",
            {},
            "SecurityError",
        ),
        (
            "Blocked import: socket",
            "import socket\ns = socket.socket()\ns.connect(('8.8.8.8', 53))",
            {},
            "SecurityError",
        ),
        (
            "Blocked import: requests",
            "import requests\nrequests.get('https://example.com')",
            {},
            "SecurityError",
        ),
        (
            "Blocked: __import__",
            "os = __import__('os')\nprint(os.getcwd())",
            {},
            "SecurityError",
        ),
        (
            "Runtime error is captured",
            "1 / 0",
            {},
            "ZeroDivisionError",
        ),
    ]

    passed = 0
    for label, code, init_state, expected in tests:
        result = client.run(code, init_state)
        combined = result.stdout + result.stderr
        ok = expected in combined
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {label}")
        if not ok:
            print(f"       expected '{expected}' in: {combined[:120]!r}")
        passed += ok

    print(f"\n{passed}/{len(tests)} passed")
