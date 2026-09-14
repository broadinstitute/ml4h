#!/usr/bin/env bash

set -e

TOOL_DIR="tools/UKB-Git-Audit-Tool"

cleanup() {
    if [ -d "$TOOL_DIR" ]; then
        echo "Removing temporary UKB Git Audit Tool..."
        rm -rf "$TOOL_DIR"
    fi

    rmdir tools 2>/dev/null || true
}

trap cleanup EXIT

echo "Cloning UKB Git Audit Tool..."
mkdir -p tools

git clone \
  https://github.com/UK-Biobank/UKB-Git-Audit-Tool.git \
  "$TOOL_DIR"

echo "Installing dependencies..."
python -m pip install -r "$TOOL_DIR/requirements.txt"

echo "Running UKB Git Audit Tool..."

printf "1\n5\n" | python "$TOOL_DIR/src/main.py"