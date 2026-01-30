#!/bin/bash
set -e

scversion="stable"

if [ -d "shellcheck-${scversion}" ]; then
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

if ! [ -x "$(command -v shellcheck)" ]; then
    if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
        echo "Please install shellcheck: https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing"
        exit 1
    fi

    # automatic local install if linux x86_64
    mkdir -p "shellcheck-${scversion}"
    curl -sL "https://github.com/koalaman/shellcheck/releases/download/${scversion}/shellcheck-${scversion}.linux.x86_64.tar.xz" -o shellcheck.tar.xz
    
    # Try xz first, fall back to unxz or python
    if command -v xz &> /dev/null; then
        xz -d shellcheck.tar.xz && tar -xf shellcheck.tar
    elif command -v unxz &> /dev/null; then
        unxz shellcheck.tar.xz && tar -xf shellcheck.tar
    elif command -v python3 &> /dev/null; then
        python3 -c "import lzma, tarfile; tarfile.open(fileobj=lzma.open('shellcheck.tar.xz')).extractall()"
    else
        echo "Error: No xz decompressor found. Please install xz-utils or shellcheck directly."
        exit 1
    fi
    rm -f shellcheck.tar shellcheck.tar.xz
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

# TODO - fix warnings in .buildkite/scripts/hardware_ci/run-amd-test.sh
find . -name "*.sh" ".git" -prune -not -path "./.buildkite/scripts/hardware_ci/run-amd-test.sh" -print0 | xargs -0 -I {} sh -c 'git check-ignore -q "{}" || shellcheck -s bash "{}"'