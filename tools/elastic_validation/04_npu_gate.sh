#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# N0/N1 hardware gates. These two qualify the assumptions every elastic
# resize is built on, so run them before the functional ladder.
#
#   N0 stateless_hccl_pg    - standby process groups can be created outside
#                             torch.distributed's global world (the reason the
#                             stock rendezvous had to be patched)
#   N1 eep_group_lifecycle  - the real elastic-EP standby-group lifecycle over
#                             a 2 -> 4 DP scale-up

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYBIN="${PYTHON:-python3}"
N0_WORLD_SIZE="${N0_WORLD_SIZE:-4}"

if ! "${PYBIN}" -c "import torch; raise SystemExit(0 if torch.npu.is_available() else 1)" \
    >/dev/null 2>&1; then
    echo "SKIP: no Ascend device visible (torch.npu.is_available() is false)"
    exit 0
fi

available=$("${PYBIN}" -c "import torch;print(torch.npu.device_count())")
if [ "${available}" -lt "${N0_WORLD_SIZE}" ]; then
    echo "SKIP: N0_WORLD_SIZE=${N0_WORLD_SIZE} but only ${available} NPUs visible"
    echo "      lower N0_WORLD_SIZE (even, >=2) or run on the 4+ device host"
    exit 0
fi

echo "== N0: stateless HCCL process groups (world_size=${N0_WORLD_SIZE}) =="
N0_WORLD_SIZE="${N0_WORLD_SIZE}" "${PYBIN}" -m pytest -sv -ra \
    tests/npu/test_stateless_hccl_pg.py

echo
echo "== N1: EEP standby-group lifecycle 2 -> 4 =="
"${PYBIN}" -m pytest -q -ra tests/npu/test_eep_group_lifecycle.py

echo
echo "N0/N1 gates: PASS - the functional ladder may proceed"
