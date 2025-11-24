#!/bin/bash
set -e

# Prepare host modules and udev triggers
depmod -a "$(uname -r)" 2>/dev/null || true
udevadm control --reload 2>/dev/null || true
udevadm trigger 2>/dev/null || true

# Install RDMA packages (Oracle Linux repos are already configured in Dockerfile)
dnf makecache \
&& dnf install -y \
    rdma-core \
    librdmacm \
    libibverbs \
    libibverbs-utils \
    infiniband-diags \
    pciutils \
    kmod \
&& dnf clean all 

# Execute user-specified command
exec "$@"
