# Elastic AFD on Ascend NPU — recipe and validation record

Status: the resize ladder below was validated on a single-machine 16x
Ascend 910 cluster with DeepSeek-V2-Lite (W8A8) on the vLLM v0.26 /
MRV1 stack (40 soak cycles, 160 resizes, no leaks; four rungs under
8-way concurrent load with drain/503 semantics). The elastic code was
since ported to ModelRunnerV2, building on the elastic-EP platform
groundwork submitted to vllm-ascend. Two version facts belong in the open
here, because the evidence below does not cover them:

1. The repository pins vLLM 0.32.0 at `5df63a72` and vLLM-Ascend at
   `bd42eea0` ([installation](../installation.md)), but the elastic patch
   seams carry `Re-anchor main@544f182d` markers. That main commit and the
   pinned release commit are on diverged lines, so the re-anchoring against
   the pinned pair is unfinished work, not a completed step.
2. Elastic AFD on NPU runs **eager only**. Graph modes are refused at
   startup: the N:1 fanout publishes one uniform row count per step taken from
   the runtime DP metadata, which a captured graph cannot bound, so an NPU
   resize in graph mode fails inside re-capture -- and because that step is the
   one that restores the Attention runtime, the failure cannot be rolled back.
3. Hardware revalidation on the ported stack was run on 2026-11-16/17; see
   "Validation record (2026-11-16)" below. The 2026-10-22 record above it is
   the pre-port v0.26/MRV1 evidence and is kept for that reason.

The device pair actually used for item 3 is vLLM-Ascend `0cc580bb9`
(PR#20321 head), which **contains** the `bd42eea0` pinned in
[installation](../installation.md); the CANN on that host was 9.1.0, not the
9.2 the installation table lists as tested.

This document covers the NPU-specific behavior of the AFD elastic resize
(`/scale_elastic_ep`). The orchestration flow, API semantics, and drain
boundary are shared with the GPU build and are documented in
[ELASTIC_AFD.md](../ELASTIC_AFD.md).

## Launch

```bash
# 1. Ray head with NPU resource detection
ray start --head ...  # ASCEND_RT_VISIBLE_DEVICES pins the visible chips

# 2. Attention-role API server (example: 2A2F eager)
vllm serve /path/to/DeepSeek-V2-Lite-W8A8 \
  --worker-cls afd_plugin.v1.worker.npu.AFDNPUAttentionWorker \
  --enable-expert-parallel --enable-elastic-ep \
  --elastic-ep-max-dp-size 4 \
  --data-parallel-backend ray --distributed-executor-backend uni \
  --data-parallel-size 2 --tensor-parallel-size 1 \
  --enforce-eager --max-model-len 4096 --max-num-seqs 16 \
  --max-num-batched-tokens 16 --gpu-memory-utilization 0.2 \
  --additional-config '{"afd": {"role": "attention",
      "connector": "CAMP2pAFDConnector", "host": "127.0.0.1",
      "port": 6241, "num_attention_ranks": 2, "num_ffn_ranks": 2,
      "compute_gate_on_attention": false},
    "afd_elastic": {"ffn_tensor_parallel_size": 1}}'
```

`--elastic-ep-max-dp-size` is required, not optional. The elastic setup raises
without it, because native two-phase scaling uses it as the Attention DP bound;
size it to the largest Attention pool the ladder reaches (4 for the 2 -> 4 -> 2
rungs below). A resize request above it must return 400 without disturbing the
running service.

## Resize

```bash
curl -X POST http://127.0.0.1:8000/scale_elastic_ep \
  -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size": 4, "role": "attention", "drain_timeout": 120}'
curl -X POST http://127.0.0.1:8000/scale_elastic_ep \
  -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size": 1, "role": "ffn", "drain_timeout": 120}'
```

## NPU-specific behavior and fixes

### CAMP2p uniform row-count protocol

The a2e/e2a kernels move `batch_size / (attn_size / ffn_size)` rows to or
from **every** Attention rank's shared-memory window; they cannot carry
unequal per-rank counts. Two rules follow, and both are enforced in Python:

1. `send_attn_output` pads every Attention rank up to the largest token
   count across all Attention ranks in the step (extra rows are dropped on
   the return path). Decode steps are usually already uniform, so padding
   cost lands on unbalanced prefill batches only.
2. The FFN side sizes everything to the same published count: the a2e/e2a
   `batch_size`, the MC2 `xActiveMask` (its dim0 must equal `x` dim0, or
   `npu_moe_distribute_dispatch_v2` fails with error 561002), and the ACL
   graph keys (`make_ffn_graph_key(..., uniform_padding=True)`), so a
   replayed FULL_DECODE_ONLY graph was captured for the batch shape the
   current step produces. Under-estimating here also makes the kernel write
   past the end of `expand_x`.

The GPU connector keeps its contiguous-group mapping; the shared graph-key
helper defaults to it and the NPU runner opts into the uniform semantics.

### Fresh FFN actors on every NPU resize

A kept FFN actor process cannot re-establish its HCCL communicators across
a world-size change: the reused rank's `hcclCommInitRootInfoConfig`
rootInfo rendezvous never meets the fresh rank's and stalls to its 180 s
timeout. `FFNActorManager.resize_actors` therefore replaces the whole FFN
actor set with fresh processes on any NPU size change (weight reload
≈15 s per rank). Attention engines are unaffected: fresh connector
generations get fresh HCCL IPC windows, and the per-launch magic counters
reset with them.

### HCCL group names are unique per connector generation

Every rebuild appends the connector port to the HCCL group names, so a
retired generation's groups never collide with the new one's. Combined
with `release_roles` before removal, this prevents HBM leaks across
repeated resizes.

## Validation record (2026-10-22, eager, DSV2Lite 2A2F)

Baseline completions at `temperature=0`, then resize, then the same
probes. Post-resize outputs were **byte-identical to baseline** at every
rung:

| Rung | Result |
| --- | --- |
| A 2→4 | HTTP 200, outputs identical, coherent |
| A 4→2 | HTTP 200, outputs identical, coherent |
| F 2→1 | HTTP 200, outputs identical, coherent |
| F 1→2 | HTTP 200, outputs identical, coherent |

Root causes found and fixed on the way (all three reproduce without the
fixes and disappear with them):

1. **Row misplacement after resize** — kernels even-split per source; the
   Python side passed raw sums. Symptom: confidently-wrong completions
   that differed across identical requests. Fixed by the uniform
   publication protocol above.
2. **F 1→2 hang** — reused FFN actor's HCCL rendezvous never completed
   ("No rank in the communicator can connect to the root node"). Fixed by
   the fresh-actors policy.
3. **A 2→4 MoE dispatch failure** — `xActiveMask dim0 != x dim0`
   (`aclnnMoeDistributeDispatchV4`, error 561002): the mask was sized to
   real tokens while `x` carried uniform-padded rows; sibling ranks also
   under-allocated `expand_x` and hung in vector-core spin-wait. Fixed by
   sizing the FFN runner's token counts (mask + graph keys) to the
   published uniform rows.

## Validation record (2026-11-16, eager, DSV2Lite, single-node 8x 910_93)

Stack: vLLM tag **v0.32.0** (`5df63a72`, editable), vLLM-Ascend `0cc580bb9`
(PR#20321 head), CANN 9.1.0, `--enforce-eager`, `--elastic-ep-max-dp-size 8`,
`--kv-cache-memory-bytes 2GiB`, booted 2A x 2F on devices 0-3.

The criterion here is **not** equality with the pre-resize baseline. An FFN
resize changes where the experts live and how many rows each rank receives, so
MC2 chunking and W8A8 accumulation order change and greedy decoding can pick a
different continuation. What must hold is that a resized deployment matches the
**same topology booted statically**. The 2026-10-22 record claims byte-identity
against the baseline for FFN rungs too; on this stack that is false for
`FFN 2->1`, and the criterion above is what the rungs are judged by now.

| Check | Result |
| --- | --- |
| N0 stateless HCCL process groups | passed, 85s |
| N1 EEP standby-group lifecycle 2->4 | passed, 88s, 7 stages x 4 ranks |
| Attention 2->4 | 200, window 74s; 3/3 probes byte-identical to baseline `00923c4f...` |
| Attention 4->2 | 200, window 9s; byte-identical |
| FFN 2->1 | 200, window 69s; differs from the 2A2F baseline, **stable across 3 probes, and identical to a static 2A1F boot** (`ecf2a9ea...` from three separate visits: first resize, second visit after 1->2->1, and the static control boot) |
| FFN 1->2 | 200, window 94s; back to the baseline digest |
| Attention 9 (> max dp 8) | 400 with the divisibility message; service healthy afterwards |
| Attention 3 (non-divisible) | 400; service healthy afterwards |
| Attention 4 with too few devices | 500 from the native store timeout (`/eep_ready/2,3`). Measured on a build that still carried a plugin-side rollback, which restored the topology and kept the deployment serving. **That rollback is no longer in the tree** (see "What happens when a resize fails"), and a pre-gate capacity check now refuses this case before any of it happens |
| FFN resize under `FULL_DECODE_ONLY` | **failed twice**; 500 from dynamo ("Pending unbacked symbols"), and unrecoverable because re-capture is the step that restores the Attention runtime. This is what the eager-only rule above encodes, and startup now refuses the combination outright |

Window lengths measured here (Attention up 74s, Attention down 9s, FFN 69s and
94s) supersede the "~10s for FFN" figure quoted in the benefit discussion: a
kept FFN actor cannot re-establish its HCCL communicators, so every NPU size
change rebuilds the actor pool and reloads weights (~15s per rank).

## What happens when a resize fails

Requests that cannot be satisfied are refused **before the scaling gate is even
set**, so a rejected request costs nothing: nothing is paused, nothing drains,
no 503 window opens. That covers the divisibility rule, the
`--elastic-ep-max-dp-size` bound, the native expert-redundancy minimum, and
device capacity (a target topology whose A+F ranks exceed the devices the Ray
cluster advertises). The capacity check deliberately fails open when the
cluster reports nothing usable, because a missing report must not refuse a
request that would otherwise have worked.

A failure *after* `prepare()` is a different matter and is not recoverable from
the plugin. Elastic EP runs the scale-up as a two-phase handshake whose first
phase creates the new DP group asynchronously inside EngineCore; the client's
timeout does not cancel it, so an unsatisfiable request leaves an in-flight
group creation running against a topology the plugin has already put back. On
2026-11-17 this was reproduced on 910_93: an `attention 4` request against a
2A2F deployment on four devices timed out after the drain, the deployment kept
answering correctly for about half an hour, and then one EngineCore exited with
`torch.distributed.DistStoreError: wait timeout after 1800000ms` from inside
`ElasticEPState._ensure_new_dp_group` (`self._prepare_future.result()`) -- the
abandoned future failing on its own clock. A rollback of the
plugin-owned state (configs, F actor pool, links, paused schedulers) was
implemented and measured, and it did not make this safe -- it converted
"immediately dead" into "dead later" -- so it was removed again rather than
shipped half-done.

**Operational rule:** if a resize returns an error, restart the deployment. The
proper fix is an abort path for the in-flight elastic EP scale-up upstream,
which this repository does not own.

## Idle deployments die after 30 minutes

Independent of resizing, and independent of elastic scaling: **an AFD
deployment that receives no traffic for more than 30 minutes stops working.**

The FFN connector loop waits in a control-plane `recv` on the `p2p` process
group, which the connector creates with a 30-minute timeout
(`afd_plugin/connectors/npu/camp2p/connector.py`). Once every Attention engine
has paused its wave, nobody publishes DP metadata any more, so that wait is
never satisfied, and when it expires the timeout is raised *inside* the FFN
loop:

1. `AFD NPU FFN worker loop failed` (`Timed out waiting 1800000ms for recv
   operation to complete`), the loop thread exits;
2. the FFN EngineCore polls the stored error (`raise_ffn_loop_error_if_any`)
   and re-raises it, so the engine core dies;
3. its teardown calls `connector.close()`, which destroys the process groups
   the surviving ranks are still using;
4. the next time traffic arrives, an Attention engine blocks and then reports
   `Connection closed by peer`, and other ranks hang in a collective forever.

Measured on 910_93, static 4A x 2F with elastic enabled: last request
06:26:37, both FFN loops timed out at 06:56:37 (5 ms apart), Attention reported
`Connection closed by peer` at 07:56:47. `/health` answers 200 throughout, so a
deployment in this state looks healthy and simply never completes a request,
which is why it is easy to mistake for an elastic defect.

Nothing in that chain is elastic-specific -- the timeout lives in connector
group setup, the CUDA connector sets the same 30 minutes, and neither the
connector nor the Attention publish path reads elastic state (the only
occurrence of "elastic" in `camp2p/connector.py` is a comment) -- so it is a
pre-existing plugin-wide property, not something this recipe introduced or can
fix on the side. **Operational rule:** keep a real completion (not `/health`)
arriving more often than every 30 minutes, or restart. `--health-check` style
liveness probes do not count: they never reach the engines.

Do not try to demonstrate that with a flag-off control, because **this recipe
cannot boot without `--enable-elastic-ep` on Ascend with the Ray DP backend**: it
dies during device assignment with `Invalid device ID` / `value 0 for parameter
userDevId is invalid. Expected value: [0, 0)`. Resolving Ray's `NPU` accelerator
resource lives in `afd_plugin/elastic/ray_actor.py`
(`AFDEngineCoreActor._set_visible_devices`), which the stock
`DPMoEEngineCoreActor` does not have. The control that works is the positive
one: on an elastic boot, keep a completion arriving every few minutes and the
deployment survives past the 30-minute mark. That control was run on 2026-11-19
on 910_93: an elastic 2A x 2F deployment, probed once every 8 minutes for 8
rounds (64 minutes, more than twice the timeout window) answered all 8, never
logged a recv timeout, and returned **byte-identical completions in all 8
rounds** (`fb264aa481c9b783` over the text field alone -- hash the text, not the
response body, which carries a random `id` and a `created` timestamp and so is
never reproducible). The gap is the trigger, not elastic and not elapsed
uptime.

### Two unrelated 1800-second timers

Everything above fires 1800s after something, and so does the failed-resize case
in the previous section, which makes them easy to confuse from the outside. Tell
them apart by the first exception line, not by the clock:

| First line | Raised from | Needs | Ours to fix |
| --- | --- | --- | --- |
| `DistStoreError: wait timeout after 1800000ms` | `ElasticEPState._ensure_new_dp_group` → `_prepare_future.result()` in an **Attention** engine | a resize that failed after `prepare()` | no -- upstream abort path |
| `Timed out waiting 1800000ms for recv operation to complete` | the **FFN** connector worker loop | silence alone, no resize ever requested | no -- pre-existing connector behaviour |

The second row needs neither a resize nor elastic enabled, which is why an idle
gap during elastic validation can manufacture what looks like a resize defect:
check the line before assigning blame to either.

## Remaining work

- Graph modes on NPU. Attempted on 2026-11-16 and refused by dynamo, so
  startup now rejects the combination. Supporting it means making the published
  row count static per capture bucket -- `_use_camp2p_dp_padding` already
  computes the group padding outside the traced forward but is gated on
  `cg_mode == NONE` -- and accepting that every Attention rank then replays the
  group maximum, which is a real throughput cost under unbalanced decode.
- Eager DBO / DBO+graph rungs.
- Concurrent traffic during resize, repeated resize soak, and
  memory/communicator leak checks across many generations.
- Mixed-role chain `2A1F → 4A1F → 4A2F → 2A2F → 2A1F`.
