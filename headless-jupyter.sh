#!/usr/bin/env zsh
set -eu

cd -- "${0:A:h}"
exec uv run --locked jupyter lab \
    --no-browser \
    --ip=127.0.0.1 \
    --port=8888 \
    --ServerApp.port_retries=0 \
    "$@"
