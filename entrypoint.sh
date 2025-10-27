#!/bin/bash
set -e

# Prepare host modules and udev triggers
depmod -a "$(uname -r)" 2>/dev/null || true
udevadm control --reload 2>/dev/null || true
udevadm trigger 2>/dev/null || true

# Execute user-specified command
exec "$@"
