#!/usr/bin/env bash
set -euo pipefail
[ -n "${REBEL_COMPILER_VERSION:-}" ] || exit 0

creds="${UV_INDEX_REBELLIONS_USERNAME}:${UV_INDEX_REBELLIONS_PASSWORD}"
host="${REBEL_PYPI_ENDPOINT%/}"
index="https://${creds}@${host#https://}/simple"

echo "+++ :package: override rebel-compiler==${REBEL_COMPILER_VERSION}"
# --python .venv on both: bare `uv pip` targets VIRTUAL_ENV (/opt/venv in the
# devtools image), not the .venv the tests run in.
uv pip uninstall --python .venv rebel-compiler
uv pip install --python .venv --extra-index-url "$index" "rebel-compiler==${REBEL_COMPILER_VERSION}"

installed="$(uv run --no-sync python -c 'import importlib.metadata as m; print(m.version("rebel-compiler"))')"
test "${installed}" = "${REBEL_COMPILER_VERSION}"
buildkite-agent annotate --style success --context rebel-compiler \
  "rebel-compiler overridden to \`${installed}\` (pypi.rebellions.in/simple)" || true
