#!/usr/bin/env bash
set -euo pipefail
[ -n "${REBEL_COMPILER_VERSION:-}" ] || exit 0

creds="${UV_INDEX_REBELLIONS_USERNAME}:${UV_INDEX_REBELLIONS_PASSWORD}"
host="${REBEL_PYPI_ENDPOINT%/}"
index="https://${creds}@${host#https://}/simple"

echo "+++ :package: override rebel-compiler==${REBEL_COMPILER_VERSION}"
# Bare `uv pip install` targets VIRTUAL_ENV (the devtools image bakes /opt/venv),
# while the tests run through `uv run` in the project .venv. Pin the target.
uv pip install --python .venv --extra-index-url "$index" "rebel-compiler==${REBEL_COMPILER_VERSION}"

installed="$(uv run --no-sync python -c 'import importlib.metadata as m; print(m.version("rebel-compiler"))')"
if [ "${installed}" != "${REBEL_COMPILER_VERSION}" ]; then
  echo "rebel-compiler in the test env is ${installed}, not ${REBEL_COMPILER_VERSION}" >&2
  exit 1
fi
buildkite-agent annotate --style success --context rebel-compiler \
  "rebel-compiler overridden to \`${installed}\` (pypi.rebellions.in/simple)" || true
