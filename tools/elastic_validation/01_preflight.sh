#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Preflight for the elastic AFD validation run.
#
# Fails on anything that would make a later result meaningless (wrong
# interpreter, missing entry point, no device). Prints WARN, not FAIL, for
# version drift against docs/installation.md, because validation environments
# legitimately differ.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYBIN="${PYTHON:-python3}"
SOC="${SOC_VERSION:-ascend910_9391}"
fail=0

ok() { printf '  ok   %s\n' "$*"; }
warn() { printf '  WARN %s\n' "$*" >&2; }
bad() {
    printf '  FAIL %s\n' "$*" >&2
    fail=1
}

echo "== 1. interpreter (project range 3.11-3.13) =="
if "${PYBIN}" - <<'PY'
import sys
print(f"  python={sys.version.split()[0]}")
sys.exit(0 if (3, 11) <= sys.version_info < (3, 14) else 1)
PY
then
    ok "interpreter in range"
else
    bad "${PYBIN} is outside 3.11-3.13; tests/unit cannot even import StrEnum"
fi

echo "== 2. elastic endpoint entry point =="
# The route is registered through [project.entry-points."vllm.endpoint_plugins"].
# Without an installed distribution (pip install -e .) vLLM never sees it and a
# resize request 404s for the wrong reason.
if "${PYBIN}" - <<'PY'
import importlib.metadata as md

found = [
    ep.value
    for ep in md.entry_points(group="vllm.endpoint_plugins")
    if "afd" in ep.name
]
print(f"  endpoint_plugins={found}")
raise SystemExit(0 if any(
    v == "afd_plugin.elastic.api:AFDElasticEndpointPlugin" for v in found
) else 1)
PY
then
    ok "AFDElasticEndpointPlugin discoverable"
else
    bad "entry point missing - run the editable install (see README step 0)"
fi

echo "== 3. runtime imports =="
for mod in vllm vllm_ascend torch torch_npu msgspec; do
    if "${PYBIN}" -c "import ${mod}" >/dev/null 2>&1; then
        version=$("${PYBIN}" -c "import ${mod};print(getattr(${mod},'__version__','?'))" 2>/dev/null || echo "?")
        ok "${mod} ${version}"
    else
        bad "cannot import ${mod}"
    fi
done

echo "== 3b. elastic endpoint module imports against this vLLM =="
# api.py does `from vllm.entrypoints.serve.elastic_ep.api_router import
# scale_elastic_ep` and overwrites the native route. If that upstream path
# moved, the whole endpoint plugin dies at import and the route never appears,
# which looks identical to a missing entry point from the outside.
if "${PYBIN}" -c "import vllm" >/dev/null 2>&1; then
    if "${PYBIN}" - <<'PY'
from afd_plugin.elastic.api import AFDElasticEndpointPlugin

print("  ELASTIC_API_OK")
PY
    then
        ok "afd_plugin.elastic.api importable"
    else
        bad "elastic api import failed - check vllm.entrypoints.serve.elastic_ep" \
            ".api_router exists in this vLLM (pin: 0.32.0 @5df63a72)"
    fi
else
    warn "vllm absent, cannot check the elastic api import"
fi

echo "== 4. pinned targets from docs/installation.md (warn only) =="
"${PYBIN}" - <<'PY'
pins = {"vllm": "0.32.0", "torch": "2.10.0", "torch_npu": "2.10.0.post7"}
import importlib.metadata as md

for name, want in pins.items():
    try:
        got = md.version(name)
    except md.PackageNotFoundError:
        continue
    print(f"  {'WARN' if not got.startswith(want) else 'ok  '} {name} got={got} pin={want}")
PY

echo "== 5. Ascend device visibility =="
if command -v npu-smi >/dev/null 2>&1; then
    count=$(npu-smi info -l 2>/dev/null | grep -c 'NPU ID' || true)
    [[ "${count}" =~ ^[0-9]+$ ]] || count=0
    if [ "${count}" -ge 4 ]; then
        ok "npu-smi sees ${count} devices"
    else
        warn "npu-smi sees ${count} devices; N0/N1 need 4, the ladder needs" \
            "the pool size you booted"
    fi
else
    warn "npu-smi not on PATH (CPU-only host? device steps will be skipped)"
fi

echo "== 6. native AFD ops build (SOC_VERSION=${SOC}) =="
ops_err=$(mktemp)
if "${PYBIN}" - 2>"${ops_err}" <<'PY'
from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded

ensure_afd_ascend_ops_loaded()
print("  AFD_OPS_OK")
PY
then
    ok "native ops loaded"
else
    bad "native ops failed: $(tail -2 "${ops_err}" | tr '\n' ' ')"
    echo "       rebuild with AFD_BUILD_ASCEND_OPS=1 SOC_VERSION=${SOC}"
fi
rm -f "${ops_err}"

echo
if [ "${fail}" -ne 0 ]; then
    echo "PREFLIGHT: FAILED (fix the FAIL lines before running the suites)"
    exit 1
fi
echo "PREFLIGHT: PASS"
