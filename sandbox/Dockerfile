# sandbox/Dockerfile
#
# Hardened Python sandbox for CodeAct agent.
# Security layers:
#   1. Non-root user (no sudo, no privilege escalation)
#   2. Read-only root filesystem (no writes anywhere)
#   3. tmpfs /tmp only (limited writes, wiped per run)
#   4. --network none (zero network access at runtime)
#   5. --memory 128m --cpus 0.5 (resource limits)
#   6. Runner itself does AST-level import blocking
#
# Usage (called by sandbox_client.py, not by hand):
#   docker run --rm \
#     --network none \
#     --read-only \
#     --tmpfs /tmp:size=16m,noexec \
#     --memory 128m --memory-swap 128m \
#     --cpus 0.5 \
#     --user sandbox \
#     finsight-sandbox
#
# The caller pipes JSON to stdin, reads JSON from stdout.

FROM python:3.13-slim

# Non-root user — no sudo, no privilege escalation
RUN useradd --no-create-home --shell /bin/false --uid 10001 sandbox

# Install only what the sandbox runner needs
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy runner, make it read-only (sandbox user can execute but not modify)
WORKDIR /app
COPY runner.py /app/runner.py
RUN chmod 444 /app/runner.py

# Drop to non-root
USER sandbox

# -u = unbuffered I/O   -I = isolated mode (ignore env, no user site-packages)
CMD ["python3", "-u", "-I", "/app/runner.py"]