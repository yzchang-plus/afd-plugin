#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# CPU unit suites, in both dependency profiles the project requires.
#
# Profile A must pass on a host WITHOUT torch or vLLM: AGENTS.md makes
# "tests/unit collects on a CPU-only host" a hard boundary, and the elastic
# suites added by this branch are the newest risk against it.
# Profile B adds the vllm_runtime-gated tests, including the CAMP2P token
# counts fixed during the rebase.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYBIN="${PYTHON:-python3}"

# Some Ascend containers export http_proxy for the whole session, and urllib
# then sends 127.0.0.1 through it: tests/unit/ffn_faas/test_service.py starts a
# local HTTP server and gets 504 Gateway Time-out instead of the 503 the phase
# guard raises. Not a code failure - keep loopback off the proxy.
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"

run_pytest() {
    "${PYBIN}" -m pytest "$@"
}

echo "== profile A: dependency-light (expect zero collect errors) =="
run_pytest tests/unit -m "not gpu and not vllm_runtime" -q -ra

echo
echo "== profile B: full unit suite (needs torch + vLLM installed) =="
if "${PYBIN}" -c "import torch, vllm" >/dev/null 2>&1; then
    run_pytest tests/unit -m "not gpu and not npu" -q -ra
else
    echo "SKIP: torch/vllm not importable here"
fi

echo
echo "== targeted: files touched by the rebase and the findings it closed =="
run_pytest -q -ra \
    tests/unit/v1/worker/test_npu_camp2p_token_counts.py \
    tests/unit/connectors/test_camp2p_token_ids.py \
    tests/unit/elastic \
    tests/unit/v1/worker/test_npu_runtime.py \
    tests/unit/v1/worker/test_runtime_classpaths.py

echo
echo "CPU unit suites: PASS"
echo "Read the token-ids result with care: test_send_attn_output_pads_routing_"
echo "to_the_uniform_row_count was written on a host without torch, vLLM or"
echo "Python 3.11, so this is its first execution. A failure there is evidence"
echo "about the production code, not a reason to loosen the assertion."
