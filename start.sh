#!/usr/bin/env bash
set -e

# Path to pipenv (which pipenv should return the correct path)
PIPENV_BIN=$(command -v pipenv || echo pipenv)

# Hugging Face cache location (shared location for downloaded models)
export TRANSFORMERS_CACHE=/opt/hf_cache
export HF_HOME=/opt/hf_cache

# Go to repo directory (location of this script)
REPO_DIR="$(cd -- "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# Ensure cache and logs directories exist
mkdir -p /opt/hf_cache
mkdir -p logs

# Run Gradio app via pipenv and append stdout/stderr to log file
# Using exec so the script PID is replaced by python process (useful if you later daemonize)
exec "$PIPENV_BIN" run python gradio_app.py >> logs/gradio.log 2>&1