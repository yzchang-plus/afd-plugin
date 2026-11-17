# Elastic AFD validation scripts

**Temporary.** These scripts exist to validate the elastic branch rebased onto
`rfc/elastic_scaling`. They are a separate commit at the tip of the branch so
they can be dropped in one step once validation passes, and they must not be
merged:

```bash
git reset --hard HEAD~1     # drops only the validation-scripts commit
```

They own no production behavior and are not part of `tests/`; the permanent
suites live in `tests/unit/elastic`, `tests/npu`, and `tests/e2e`.

## Order of operations

| Step | Script | Needs | Proves |
| --- | --- | --- | --- |
| 0 | install | CANN env | see below, the editable install is not optional |
| 1 | `01_preflight.sh` | any host | interpreter in 3.11-3.13, endpoint entry point installed, runtime imports, ops build, device count |
| 2 | `02_static.sh` | `uv sync --group dev` | ruff, mypy, shellcheck, SPDX, markdownlint over the PR range |
| 3 | `03_cpu_unit.sh` | step 2 | both CPU profiles, plus the files the rebase touched |
| 4 | `04_npu_gate.sh` | 4 NPUs | N0 stateless HCCL groups, N1 EEP group lifecycle |
| 5 | `05_elastic_ladder.sh` | running server | the resize ladder, byte-identical outputs, clean rejections |
| 6 | `06_known_p1_repro.sh` | GPU build (+graph serve script) | gate check: graph+elastic refused at startup; CUDA warm pairing tripwire |

Run them in that order: a failure at step 1 or 4 makes the step 5 result
meaningless.

## Step 0

```bash
uv sync --group dev                    # the branch added the msgspec dependency
AFD_BUILD_ASCEND_OPS=1 SOC_VERSION=ascend910_9391 \
  python -m pip install -v --no-build-isolation --no-deps -e .
```

The editable install is what registers
`[project.entry-points."vllm.endpoint_plugins"] afd_elastic`. Without it
vLLM never loads `AFDElasticEndpointPlugin` and `/scale_elastic_ep` returns
404 for a reason that has nothing to do with elastic scaling. `01_preflight.sh`
checks exactly this.

## Full validation order (with the device budget)

Device arithmetic matters here: a topology of `A x F` needs `A + F` devices at
TP=1, and the capacity preflight now refuses a scale-up that needs more devices
than the deployment can schedule. So the eager ladder that reaches attention 4
must be booted with at least 6 visible devices, and the deliberate-failure probe
needs a boot that is intentionally too small.

```bash
# 1. tree first, server second (rule 1 above)
git log --oneline -1
AFD_BUILD_ASCEND_OPS=0 python -m pip install -v --no-build-isolation --no-deps -e .
python -c "import afd_plugin; print(afd_plugin.__file__)"      # must be this clone

# 2. static + CPU gates
bash tools/elastic_validation/01_preflight.sh
GIT_SSL_NO_VERIFY=1 PIP_NO_BUILD_ISOLATION=1 bash tools/elastic_validation/02_static.sh
bash tools/elastic_validation/03_cpu_unit.sh

# 3. device qualification BEFORE any server holds devices
bash scripts/run_npu_qual030b.sh tests/npu/test_stateless_hccl_pg.py       # N0
bash scripts/run_npu_qual030b.sh tests/npu/test_eep_group_lifecycle.py     # N1

# 4. eager happy path: 6 devices so attention 4 is actually satisfiable
CARDS=0,1,2,3,4,5 nohup bash scripts/serve_2a2f_eager030b.sh > /tmp/v6.log 2>&1 &
#    wait until a completion returns text (not /health), then:
MODEL=<served-name> MAX_DP=8 bash tools/elastic_validation/05_elastic_ladder.sh

# 5. rejection probes against the same server (each must leave it serving)
curl -s -X POST http://127.0.0.1:8000/scale_elastic_ep -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size":10,"role":"attention","drain_timeout":120}'   # max-dp, 10 divides by 2
curl -s -X POST http://127.0.0.1:8000/scale_elastic_ep -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size":3,"role":"attention","drain_timeout":120}'    # non-divisible
curl -s -m 20 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served-name>","prompt":"hi","max_tokens":1,"temperature":0}'  # must still answer

# 6. capacity preflight: needs a deliberately too-small boot
CARDS=0,1,2,3 nohup bash scripts/serve_2a2f_eager030b.sh > /tmp/v4.log 2>&1 &
#    then request attention 4 and expect an immediate 400 naming the device count,
#    followed by a completion that still returns text.

# 7. native minimum: boot 4A x 2F with num_redundant_experts=0
MODEL=<served-name> MAX_DP=8 UNDER_MIN=2 RUNGS="attention:2" bash tools/elastic_validation/05_elastic_ladder.sh

# 8. same-topology control, to judge FFN rungs (see the criterion below)
sed 's/"num_ffn_ranks": 2/"num_ffn_ranks": 1/' scripts/serve_2a2f_eager030b.sh \
  > scripts/serve_2a1f_control.sh          # keep it in scripts/, never /tmp (rule 2)
#    probe three times and compare digests with after_ffn_1_1.txt

# 9. graph gate last, with port 8000 free
pkill -TERM -f "vllm serve"; sleep 20; ray stop --force
GRAPH_SERVE=scripts/serve_2a2f_graph.sh bash tools/elastic_validation/06_known_p1_repro.sh
```

Anything that reports SKIP is uncovered, not passed.

## Running the ladder

Start the Attention-role server as described in
[`docs/npu/ELASTIC_AFD_NPU.md`](../../docs/npu/ELASTIC_AFD_NPU.md), then:

```bash
MODEL=/path/to/DeepSeek-V2-Lite-W8A8 MAX_DP=4 bash tools/elastic_validation/05_elastic_ladder.sh
```

The script never starts or stops a server: booting is topology-defining, and a
script that guessed it would validate the wrong configuration.

Knobs (all environment variables, all optional unless noted):

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL` | required | served model path, must match `vllm serve` |
| `BASE_URL` | `http://127.0.0.1:8000` | Attention-role API server |
| `RUNGS` | `attention:4 attention:2 ffn:1 ffn:2` | ladder, `role:size` pairs |
| `CYCLES` | `1` | repeat the ladder, for the soak evidence |
| `MAX_DP` | `4` | the `--elastic-ep-max-dp-size` you booted with |
| `UNDER_MIN` | unset | set `2` when booted 4A/2F with `num_redundant_experts=0` |
| `PROMPTS_FILE` | unset | one prompt per line, replaces the built-in set |
| `MAX_TOKENS` | `32` | probe length |
| `DRAIN_TIMEOUT` | `120` | passed through to the resize request |
| `SCALE_TIMEOUT` | `600` | client-side wait for the resize; keep it above the native 300s store timeout |
| `RUNG_PROBES` | `3` | repeat probes per rung; disagreement between identical requests is the row-misplacement signature |
| `ART_DIR` | `/tmp/elastic_ladder/<utc>` | where baseline and per-rung completions are kept |
| `CONTROL_DIR` | unset | same-topology control digests, as `control_<role>_<size>.txt` |

With `CONTROL_DIR` set, a rung that differs from the pre-resize baseline but
matches the same topology booted statically is reported as
`MATCHES the same-topology control ... resize is faithful` and does **not**
count as a failure. Without it the script can only report CHANGED and say it
cannot distinguish topology numerics from a wrong leftover state -- which is
the honest reading, not a pass. Produce a control file by booting the target
topology directly (no resize) and saving its probe digest:

```bash
mkdir -p /tmp/ctl
for i in 1 2 3; do python tools/elastic_validation/scale_probe.py --timeout 60 probe \
  --model <served-name> --max-tokens 32 --save /tmp/ctl/probe_$i.txt; done
cp /tmp/ctl/probe_1.txt /tmp/ctl/control_ffn_1.txt     # 2A1F static boot
# then: MODEL=<served-name> CONTROL_DIR=/tmp/ctl bash tools/elastic_validation/05_elastic_ladder.sh
```

| `GRAPH_SERVE` | `/new_data/cyz/verify030b/scripts/serve_2a2f_graph.sh` | graph variant used by step 6's gate check |
| `GRAPH_BOOT_TIMEOUT` | `300` | how long step 6 waits for the gate to fire |
| `BACKEND` | `npu` | `gpu` additionally runs the CUDA pairing check |

## What the checks actually establish

- `05` compares a SHA-256 over the probe texts, so a rung passes only if every
  completion is byte-identical to the baseline, not merely HTTP 200.
- A changed digest is not automatically a broken resize. Byte-identity to the
  *pre-resize* baseline is the invariant recorded in
  `docs/npu/ELASTIC_AFD_NPU.md` for the vLLM v0.26 / MRV1 stack; on the current
  ModelRunnerV2 target an FFN resize can legitimately change numerics (one FFN
  rank then owns every expert and takes 2x the rows, which changes MC2 chunking
  and W8A8 accumulation order). Three signatures separate the cases:
  disagreement between identical probes is misrouting; a stable post-resize text
  that differs from a same-topology control boot is a resize that left the wrong
  state; a stable post-resize text that matches the control boot is topology
  numerics, and the doc's invariant claim is what needs revisiting.
- The rejection rungs re-probe after each 400. A rejection that leaves the
  service stopped is the failure PR #523 review described, and only this
  sequence catches it.
- `UNDER_MIN` covers the native EP **minimum** bound. The `2 -> 4 -> 2` ladder
  cannot reach it: the scale-up first creates the expert redundancy that makes
  the way back legal, so a plain ladder run never exercises that guard.
- `05` samples `npu-smi info` into `elastic_hbm.log` across FFN resizes so a
  second KV budget resident during a resize is visible.
- Step 6 section A checks the startup gate: a graph-mode + elastic boot must
  die with "supports only --enforce-eager" before the engines exist. It reports
  PASS / FAIL / SKIP, and SKIP (no graph script, or port 8000 busy) is not a
  pass. Section B stays a tripwire for the CUDA warm-up pairing: `not
  reproduced` is expected and `CONFIRMED` is a regression; it needs `BACKEND=gpu`
  on a GPU host, so on Ascend it will report SKIP.

## Three rules this branch already paid for

1. **Update the tree before starting the server, never after.** The code under
   test runs *inside* `vllm serve`; a `git checkout` after the server is up
   changes the working tree but not the process, so the result belongs to an
   unknown build. Print `git log --oneline -1` before launching and record it
   with the result.
2. **Never put a launch script in `/tmp`.** The serve recipes begin with
   `cd "$(dirname "$0")"`, so a script copied to `/tmp` starts vLLM with `/tmp`
   as the working directory, where a leftover `vllm*` directory shadows the
   installed package and the engines hang before any AFD code runs. Keep derived
   variants next to the originals under `scripts/`.
3. **An idle gap of 30 minutes is a test artifact, not a result.** The FFN
   control-plane `recv` times out 1800s after the last DP metadata, which kills
   the FFN loop, whose teardown destroys the process groups every other rank is
   still using. A ladder that pauses half an hour between rungs — or a soak left
   running over a meal break — will then report a `Connection closed by peer`
   that has nothing to do with the resize being tested. Keep a completion
   arriving at least every 20 minutes during any wait (`/health` does not count;
   it never reaches the engines), and when a run does go quiet, read the actor
   logs for `Timed out waiting 1800000ms for recv` before believing anything
   about elastic. Both halves are measured: 30 minutes of silence bricked a
   static 4A×2F boot, while the same build probed every 8 minutes for 64 minutes
   answered 8/8 with no timeout logged. The mechanism is base connector
   behaviour; see `docs/npu/ELASTIC_AFD_NPU.md`, "Idle deployments die after 30
   minutes". Do not try to re-prove that with a flag-off boot — see the
   `Invalid device ID` row under "Failures that are not ours".

## Evidence comes before cleanup

A teardown command that removes a Ray session directory also removes the only
record of why an engine died. Cost of getting this wrong once: the first idle
death at 4A/2F could not be attributed at all, and only the *second* incident
supplied the log line (`Timed out waiting 1800000ms for recv operation to
complete`) that explained both.

Archive, then look, then clean — in that order, and never in the same command
as the next boot:

Do not guess which Ray temp directory holds the incident: a host can have two
(`/tmp/ray` and a `--temp-dir` the launch scripts set), and `session_latest` may
then point at the one with no engine logs in it — 29 files of raylet and
dashboard logs, and nothing from the actors. Ask a process instead:

```bash
ls -l /proc/<engine-pid>/fd/2        # where that actor's stderr really goes
D=/tmp/raykeep/$(date +%Y%m%d-%H%M%S)
mkdir -p "$D" && cp -a "$(dirname "$(readlink /proc/<engine-pid>/fd/2)")" "$D/"
```

The per-actor files are `worker-<id>-<session>-<pid>.err`, so grep them by pid.

Two rules that the archive exists to serve:

- **Who died first is the whole question.** Sort the archived errors by the
  timestamp *inside* the line, not by file order, and read from the top. gloo
  `Connection closed by peer` on the surviving ranks is a consequence: the
  shutdown path of any AFD rank calls `connector.close()`, which destroys the
  shared gloo/HCCL groups every other rank is still using (see
  `afd_plugin/v1/worker/npu/attention_model_runner_v2.py` `shutdown`, and the
  same in the FFN runners). It is base behaviour, not elastic behaviour, so it
  is *never* the first error to explain.
- **The process list can lie.** A rank whose busy loop has exited keeps its
  process alive, so `pgrep -c` still reports the expected count. Read
  `/proc/<pid>/task/*/comm` for the loop thread instead
  (`afd-npu-ffn-worker-loop`), and compare `lstart` across actors: a restarted
  actor has a newer start time than its siblings.

## Failures that are not ours

Recorded from the rebase validation so a red line does not cost another run:

| Case | Verdict | Evidence |
| --- | --- | --- |
| Deployment stops answering ~30 min after the last request; `/health` still 200 | base, not elastic | actor logs: `AFD NPU FFN worker loop failed` / `Timed out waiting 1800000ms for recv operation to complete` exactly 1800s after the last metadata, then `Connection closed by peer` on Attention. The 30-minute timeout is on a context line of this branch's diff (`camp2p/connector.py`), and `recv_dp_metadata_list` is not touched by it at all. See `docs/npu/ELASTIC_AFD_NPU.md`, "Idle deployments die after 30 minutes" |
| `test_npu_feature_validation[gate-on-attention]` | upstream, pre-existing | fails identically on `2bede02` (the branch base), same `DID NOT RAISE`. Report separately; do not attribute to this branch |
| `ffn_faas::test_readiness_and_host_stop_request` | container proxy | `http_proxy` makes `urlopen` send `127.0.0.1` through it and return 504 instead of the 503 the phase guard raises. `03_cpu_unit.sh` now exports loopback into `no_proxy`; with that it passes |
| `test_package::test_entry_point_is_registered` | environment | only when the project is not installed into the interpreter (e.g. a `pythonpath`-only venv). Red on the base as well |
| A `--enable-elastic-ep`-less variant of the serve recipe dies at boot with `Invalid device ID` / `value 0 for parameter userDevId is invalid. Expected value: [0, 0)` | expected, not a bug to chase | resolving Ray's `NPU` accelerator resource lives in `elastic/ray_actor.py::AFDEngineCoreActor._set_visible_devices`, which the stock actor does not have, so without elastic the workers get an empty visible-device list. **Consequence: do not use "boot the same build with elastic switched off" as a control** — it cannot boot on this host, and neither can the branch base. Use the keep-traffic control instead (probe every <10 min and show the deployment survives). |

## Coverage that does not exist

- The routing-row padding does have a CPU test, in
  `tests/unit/connectors/test_camp2p_token_ids.py`, named
  `test_send_attn_output_pads_routing_to_the_uniform_row_count`. It drives the
  real `send_attn_output` against recorded operator calls. It was written on a
  host with no Python 3.11, no torch and no vLLM, so it has never executed:
  step 3 is its first run, and a failure there is evidence about the code, not
  a reason to relax the assertion.
- Nothing pins the N:1 ubatching guard. The uniform publication now skips a
  step that the ubatched path resizes to its own per-group physical count, so
  the negative-length symptom cannot come from that combination any more, but
  the DBO rung itself has never run on hardware and
  [`docs/npu/ELASTIC_AFD_NPU.md`](../../docs/npu/ELASTIC_AFD_NPU.md) still
  lists it as remaining work. A `RUNGS` run against a `--num-ubatches 2` boot
  is the only evidence either way.
- Re-anchoring the elastic patch seams from `main@544f182d` onto the pinned
  vLLM 0.32.0 `5df63a72` line is unfinished. No script tests it; expect the
  drift to show up as an `AttributeError` or `TypeError` at boot rather than as
  a rejected resize.
