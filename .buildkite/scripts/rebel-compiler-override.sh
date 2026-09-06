#!/usr/bin/env bash
set -euo pipefail
[[ -n "${REBEL_COMPILER_VERSION:-}" ]] || exit 0

creds="${UV_INDEX_REBELLIONS_USERNAME}:${UV_INDEX_REBELLIONS_PASSWORD}"
host="${REBEL_PYPI_ENDPOINT%/}"
index="https://${creds}@${host#https://}/simple"

echo "+++ :package: override rebel-compiler==${REBEL_COMPILER_VERSION}"

# `uv run` resolves the interpreter the tests use; bare `uv pip` would target VIRTUAL_ENV instead.
python="$(uv run --no-sync python -c 'import sys; print(sys.executable)')"
uv pip uninstall --python "${python}" rebel-compiler
uv pip install --python "${python}" --extra-index-url "${index}" "rebel-compiler==${REBEL_COMPILER_VERSION}"

installed="$(uv run --no-sync python -c 'import importlib.metadata as m; print(m.version("rebel-compiler"))')"
[[ "${installed}" == "${REBEL_COMPILER_VERSION}" ]]

buildkite-agent annotate --style success --context rebel-compiler \
  "rebel-compiler overridden to \`${installed}\` (pypi.rebellions.in/simple)" || true
