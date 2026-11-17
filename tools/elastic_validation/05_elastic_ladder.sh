#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Functional elastic resize ladder against a running AFD service.
#
# Start the Attention-role server yourself (docs/npu/ELASTIC_AFD_NPU.md), then:
#   MODEL=/path/to/DeepSeek-V2-Lite-W8A8 ./tools/elastic_validation/05_elastic_ladder.sh
#
# What it proves, in order:
#   1. baseline completions at temperature=0
#   2. every rung of the ladder returns 200 AND leaves outputs byte-identical
#      to the baseline (the documented invariant)
#   3. over-bound and non-divisible requests return 400 and the service is
#      still healthy afterwards - a clean rejection that kills serving is
#      exactly what PR #523 review flagged
#   4. HBM usage is sampled across FFN resizes, so a second KV budget
#      resident during a resize shows up in the log

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${here}/../.."

PYBIN="${PYTHON:-python3}"
HELPER="${here}/scale_probe.py"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-}"
RUNGS="${RUNGS:-attention:4 attention:2 ffn:1 ffn:2}"
CYCLES="${CYCLES:-1}"
MAX_DP="${MAX_DP:-4}"                 # --elastic-ep-max-dp-size you booted with
UNDER_MIN="${UNDER_MIN:-}"            # set e.g. 2 when booted 4A/2F with num_redundant_experts=0
PROMPTS_FILE="${PROMPTS_FILE:-}"      # one prompt per line; a project default set otherwise
DRAIN_TIMEOUT="${DRAIN_TIMEOUT:-120}"

if [ -z "${MODEL}" ]; then
    echo "MODEL is required (the served model path, must match vllm serve)" >&2
    exit 1
fi

# Artifacts are kept on purpose: a digest mismatch is only actionable when the
# two completions that produced it can still be read side by side.
work="${ART_DIR:-/tmp/elastic_ladder/$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "${work}"
base="${work}/baseline.txt"
cur="${work}/current.txt"
watch_log="${work}/hbm.log"
# The row-misplacement signature in docs/npu/ELASTIC_AFD_NPU.md is output that
# varies between identical requests, so one probe per rung cannot separate a
# stable-but-different topology from a misrouted one.
RUNG_PROBES="${RUNG_PROBES:-3}"
# Directory holding same-topology control digests, named
# control_<role>_<size>.txt (first line = digest). A rung whose output
# differs from the pre-resize baseline but matches that control is
# reporting topology numerics, not a resize defect.
CONTROL_DIR="${CONTROL_DIR:-}"
fail=0

probe() { # save_path
    if [ -n "${PROMPTS_FILE}" ]; then
        "${PYBIN}" "${HELPER}" --base-url "${BASE_URL}" probe \
            --model "${MODEL}" --max-tokens "${MAX_TOKENS:-32}" \
            --prompts-file "${PROMPTS_FILE}" --save "$1"
    else
        "${PYBIN}" "${HELPER}" --base-url "${BASE_URL}" probe \
            --model "${MODEL}" --max-tokens "${MAX_TOKENS:-32}" --save "$1"
    fi
}

same_as_baseline() {
    [ -f "$1" ] && [ "$(head -1 "$1")" = "$(head -1 "${base}")" ]
}

same_digest() { # file_a file_b
    [ -f "$1" ] && [ -f "$2" ] && [ "$(head -1 "$1")" = "$(head -1 "$2")" ]
}

start_watch() {
    command -v npu-smi >/dev/null 2>&1 || return 0
    (
        while :; do
            printf '=== %s ===\n' "$(date -u +%H:%M:%S)"
            npu-smi info || true
            sleep 2
        done
    ) >>"${watch_log}" 2>&1 &
    watch_pid=$!
}

stop_watch() {
    [ -n "${watch_pid:-}" ] || return 0
    kill "${watch_pid}" 2>/dev/null || true
    wait "${watch_pid}" 2>/dev/null || true
    watch_pid=""
}

resize() { # role size expect label
    local role="$1" size="$2" expect="$3" label="$4" rc=0 out=""
    # SCALE_TIMEOUT must exceed the native elastic-EP store timeout (300s) plus
    # the drain window, or the client hangs up while the server is still
    # reconfiguring and every later rung reports a cascading false failure.
    out=$("${PYBIN}" "${HELPER}" --base-url "${BASE_URL}" --timeout "${SCALE_TIMEOUT:-600}" \
        scale --role "${role}" --new-size "${size}" \
        --drain-timeout "${DRAIN_TIMEOUT}" --expect "${expect}") || rc=$?
    printf '%s\n' "${out}"
    if [ "${rc}" -ne 0 ]; then
        printf '  RESULT %-28s UNEXPECTED (rc=%s)\n' "${label}" "${rc}"
        fail=1
        return 0
    fi
    printf '  RESULT %-28s as expected\n' "${label}"
}

echo "== 0. service reachable =="
if ! probe "${base}"; then
    echo "cannot get a baseline from ${BASE_URL} - is the server up and the" \
        "model path correct? (a 404 on /scale_elastic_ep later means the" \
        "endpoint entry point is not installed: see 01_preflight.sh)" >&2
    exit 1
fi
echo "baseline digest: $(head -1 "${base}")"
echo

read -r -a rung_list <<< "${RUNGS}"
for ((cycle = 1; cycle <= CYCLES; cycle++)); do
    echo "===== cycle ${cycle}/${CYCLES} ====="
    for rung in "${rung_list[@]}"; do
        role="${rung%%:*}"
        size="${rung##*:}"
        echo "-- ${role} -> ${size}"
        t0=$(date +%s)
        if [ "${role}" = "ffn" ]; then
            start_watch
            resize "${role}" "${size}" 200 "${role} ${size} (200)"
            stop_watch
        else
            resize "${role}" "${size}" 200 "${role} ${size} (200)"
        fi
        # The window length is itself a result: the PR's benefit statement quotes
        # FFN ~10s / attention 1-2min, and this is the number to check it against.
        echo "  window=$(( $(date +%s) - t0 ))s"

        first="${work}/after_${role}_${size}_1.txt"
        if ! probe "${first}"; then
            echo "  FAIL service stopped answering after ${role} -> ${size}"
            fail=1
            continue
        fi
        diverged=0
        for ((n = 2; n <= RUNG_PROBES; n++)); do
            if ! probe "${work}/after_${role}_${size}_${n}.txt"; then
                echo "  FAIL service stopped answering after ${role} -> ${size}"
                fail=1
                diverged=1
                break
            fi
            if ! same_digest "${first}" "${work}/after_${role}_${size}_${n}.txt"; then
                echo "  FAIL NON-DETERMINISTIC after ${role} -> ${size}: identical" \
                    "requests disagree (row-misplacement signature)"
                echo "       probe1=$(head -1 "${first}")"
                echo "       probe${n}=$(head -1 "${work}/after_${role}_${size}_${n}.txt}")"
                fail=1
                diverged=1
                break
            fi
        done
        [ "${diverged}" -eq 1 ] && continue
        control="${CONTROL_DIR}/control_${role}_${size}.txt"
        if same_as_baseline "${first}"; then
            echo "  outputs byte-identical to baseline"
        elif [ -n "${CONTROL_DIR}" ] && [ -f "${control}" ] \
                && same_digest "${first}" "${control}"; then
            # The documented criterion: a resized deployment must match the same
            # topology booted statically. An FFN resize moves experts and row
            # counts, so MC2 chunking and W8A8 accumulation order change and
            # greedy decoding may take a different continuation -- that is a
            # property of the topology, not of the resize.
            echo "  differs from baseline, but MATCHES the same-topology" \
                "control (${control})"
            echo "  => topology numerics, resize is faithful"
        else
            echo "  outputs CHANGED after ${role} -> ${size} but stable across" \
                "${RUNG_PROBES} probes"
            echo "       baseline=$(head -1 "${base}")"
            echo "       current =$(head -1 "${first}")"
            if [ -z "${CONTROL_DIR}" ]; then
                echo "       no CONTROL_DIR set: cannot tell topology numerics apart" \
                    "from a resize that left the wrong state"
            elif [ ! -f "${control}" ]; then
                echo "       missing control file ${control} -- boot ${role}=${size}" \
                    "statically and save its digest there"
            else
                control_digest=$(head -1 "${control}")
                echo "       DIFFERS from the control (${control_digest}):" \
                    "the resize left the wrong state"
            fi
            diff -u "${base}" "${first}" | head -40 || true
            fail=1
        fi
    done

    echo "-- rejection paths (service must survive each)"
    resize attention "$((MAX_DP + 1))" 400 "over max dp (${MAX_DP}) -> 400"
    if probe "${cur}" >/dev/null && same_as_baseline "${cur}"; then
        echo "  healthy after over-bound rejection"
    else
        echo "  FAIL service unhealthy after over-bound rejection"
        fail=1
    fi

    resize attention 3 400 "non-divisible A:F -> 400"
    if probe "${cur}" >/dev/null && same_as_baseline "${cur}"; then
        echo "  healthy after non-divisible rejection"
    else
        echo "  FAIL service unhealthy after non-divisible rejection"
        fail=1
    fi

    if [ -n "${UNDER_MIN}" ]; then
        # The case the 2 -> 4 -> 2 ladder cannot reach: the preceding scale-up
        # creates the expert redundancy that makes the way back legal, so the
        # native minimum is never exercised. Boot 4A/2F with
        # num_redundant_experts=0 and set UNDER_MIN=2 to hit it directly.
        resize attention "${UNDER_MIN}" 400 "under native EP min -> 400"
        if probe "${cur}" >/dev/null && same_as_baseline "${cur}"; then
            echo "  healthy after under-min rejection"
        else
            echo "  FAIL service unhealthy after under-min rejection"
            fail=1
        fi
    else
        echo "  SKIP under-native-min rejection (set UNDER_MIN when booted 4A/2F"
        echo "       with num_redundant_experts=0)"
    fi
done

if [ -s "${watch_log}" ]; then
    cp "${watch_log}" "${PWD}/elastic_hbm.log"
    echo
    echo "HBM samples across FFN resizes -> ${PWD}/elastic_hbm.log"
    echo "Check that no device holds two full KV budgets during a resize."
fi

echo
echo "artifacts -> ${work}"
echo "  baseline.txt, after_<role>_<size>_<probe>.txt: first line is the digest,"
echo "  the rest is the completion text. Diff a post-resize file against a"
echo "  same-topology control boot to separate topology numerics from a"
echo "  resize that leaves the wrong state."

echo
if [ "${fail}" -ne 0 ]; then
    echo "LADDER: FAILURES PRESENT"
    exit 1
fi
echo "LADDER: PASS"
