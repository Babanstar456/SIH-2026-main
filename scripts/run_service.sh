#!/usr/bin/env bash
# Bootstrap + launch for dhwanik.service.
#
# On every start: create the venv if it doesn't exist yet, install/update
# dependencies ONLY if requirements.txt has changed since the last successful
# install (a stamp file avoids re-running pip on every service restart - a
# crash-looping service would otherwise spend its restart budget on pip, not
# on actually running), then `exec` main.py so systemd tracks the real
# python process directly instead of this wrapper shell (correct signal
# delivery on stop/restart).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

VENV_DIR="$REPO_DIR/.venv"
REQ_FILE="$REPO_DIR/requirements.txt"
STAMP_FILE="$VENV_DIR/.requirements.stamp"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "[run_service] no venv at $VENV_DIR - creating one"
    python3 -m venv "$VENV_DIR"
fi

REQ_HASH="$(sha256sum "$REQ_FILE" | cut -d' ' -f1)"
if [ ! -f "$STAMP_FILE" ] || [ "$(cat "$STAMP_FILE")" != "$REQ_HASH" ]; then
    echo "[run_service] requirements.txt changed (or first run) - installing dependencies"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r "$REQ_FILE"
    echo "$REQ_HASH" > "$STAMP_FILE"
else
    echo "[run_service] dependencies already match requirements.txt - skipping install"
fi

exec "$VENV_DIR/bin/python" "$REPO_DIR/main.py" "$@"
