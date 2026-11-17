#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Regression check for the two findings PR #523 review raised, now closed on
# this branch. Read it as a tripwire: a CONFIRMED line means the fix has
# regressed or was never effective on this hardware, and "not reproduced" is
# the passing result. An earlier revision of this script documented the
# opposite, when the findings were still open.
#
# The skip paths are not passes. If GRAPH_SERVE or BACKEND is missing, the
# corresponding finding was not checked at all, so report it as not covered
# rather than as green.
#
# A - NPU, graph mode + elastic must be REFUSED AT STARTUP
#     Elastic AFD on Ascend supports eager only. The N:1 fanout publishes one
#     uniform row count per step taken from the runtime DP metadata, which
#     dynamo cannot bound: two FULL_DECODE_ONLY runs on 910_93 returned 500 from
#     re-capture during an FFN resize, and because re-capture is the step that
#     restores the Attention runtime, no rollback can recover it. validate_-
#     elastic_config therefore rejects the combination before the engines exist.
#     This section boots the graph variant and asserts it dies with that
#     message. Set GRAPH_SERVE to a script that differs from the eager one only
#     in the compilation flag.
#
# B - CUDA, attention scale-up collective pairing
#     Stock prepare_reconfiguration() posts an all-rank all_reduce over the
#     standby dp/ep groups, while the joining engine never issues the matching
#     leg (AFD A engines have no fused-MoE reuse, so warmup_new_worker skips
#     warm_and_capture). Both sides used to wait on each other and the request
#     neither returned 200 nor 400 - it hung. The fix exempts the warm on BOTH
#     sides from AFDElasticGPUExecutor (_warm_target_groups returns immediately
#     and warm_and_capture runs under skip_dp_coordination), leaving
#     prepare_new_worker a no-op by design; the first post-switch serving
#     collective then initializes the comms with all ranks present. A hang here
#     means that exemption was lost. Requires the GPU build; BACKEND=gpu.

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${here}/../.."

PYBIN="${PYTHON:-python3}"
HELPER="${here}/scale_probe.py"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
BACKEND="${BACKEND:-npu}"
SCALE_TIMEOUT="${SCALE_TIMEOUT:-600}"  # B: wait this long before calling it a hang
fail=0

echo "== A: graph mode + elastic refused at startup =="
GRAPH_SERVE="${GRAPH_SERVE:-/new_data/cyz/verify030b/scripts/serve_2a2f_graph.sh}"
GRAPH_BOOT_TIMEOUT="${GRAPH_BOOT_TIMEOUT:-300}"
gate_log=/tmp/graph_gate_boot.log
if [ ! -f "${GRAPH_SERVE}" ]; then
    echo "  SKIP GRAPH_SERVE=${GRAPH_SERVE} does not exist, so the gate was not"
    echo "       checked. Derive it from the eager script changing only the"
    echo "       compilation flag (see tools/elastic_validation/README.md)."
elif (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ':8000 '; then
    echo "  SKIP port 8000 is busy: stop the running server first, or this check"
    echo "       would test nothing (the gate boots its own instance)."
else
    timeout "${GRAPH_BOOT_TIMEOUT}" bash "${GRAPH_SERVE}" >"${gate_log}" 2>&1 &
    boot=$!
    verdict=""
    for _ in $(seq 1 40); do
        sleep 5
        if grep -q "supports only --enforce-eager" "${gate_log}"; then
            verdict=refused
            break
        fi
        if curl -s -m 5 -o /dev/null http://127.0.0.1:8000/health 2>/dev/null; then
            verdict=reachable
            break
        fi
        kill -0 "${boot}" 2>/dev/null || { verdict=exited; break; }
    done
    kill "${boot}" 2>/dev/null || true
    wait "${boot}" 2>/dev/null || true
    # This check starts a server, and the serve recipe starts a Ray head with
    # placement groups for the engines. Killing only the API server would leave
    # that cluster and its device reservations behind, which then makes the next
    # boot fail with "Not enough resources to allocate placement groups" --
    # indistinguishable from an elastic defect. Tear the whole stack down.
    pkill -TERM -f "vllm serve" 2>/dev/null || true
    sleep 10
    ray stop --force >/dev/null 2>&1 || true
    case "${verdict}" in
        refused)
            echo "  PASS: refused before the engines existed"
            grep -m1 "supports only --enforce-eager" "${gate_log}" | head -c 220
            echo
            ;;
        reachable)
            echo "  FAIL: a graph-mode server became reachable - the gate did not fire"
            fail=1
            ;;
        exited)
            echo "  FAIL: boot exited without the gate message (see ${gate_log})"
            tail -20 "${gate_log}"
            fail=1
            ;;
        *)
            echo "  INCONCLUSIVE: no verdict within ${GRAPH_BOOT_TIMEOUT}s (${gate_log})"
            ;;
    esac
fi

echo
echo "== B: CUDA attention 2 -> 4 (warmup collective pairing) =="
if [ "${BACKEND}" = "gpu" ]; then
    rc=0
    timeout "${SCALE_TIMEOUT}" "${PYBIN}" "${HELPER}" --base-url "${BASE_URL}" \
        scale --role attention --new-size 4 --drain-timeout 60 --expect 200 || rc=$?
    case "${rc}" in
        0)
            echo "  not reproduced: scale-up returned 200"
            ;;
        1 | 124)
            echo "  CONFIRMED: no 200 within ${SCALE_TIMEOUT}s - unmatched" \
                "warmup collective (B)"
            fail=1
            ;;
        *)
            echo "  inconclusive: rc=${rc}; read the status line above"
            ;;
    esac
else
    echo "  SKIP needs the GPU build (set BACKEND=gpu)"
fi

echo
if [ "${fail}" -ne 0 ]; then
    echo "KNOWN-P1: REGRESSED - a finding this branch closes reproduced anyway."
    echo "           Re-read the section header: the configuration may also have"
    echo "           failed to exercise the path, which is not a pass either."
    exit 1
fi
echo "KNOWN-P1: no regression observed."
echo "           A section that printed SKIP is not covered, not passed."
