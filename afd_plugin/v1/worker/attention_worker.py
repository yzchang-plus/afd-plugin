# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side worker for AFD GPU execution."""

from __future__ import annotations

from copy import deepcopy

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner as gpu_model_runner_v2
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.worker_base import CompilationTimes

from afd_plugin.elastic.config import is_elastic_attention
from afd_plugin.elastic.gpu import (
    AFDElasticGPUExecutor,
    connect,
    finish_attention,
    release_link,
    send_stop,
    update_topology,
)
from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.attention_model_runner_v2 import (
    AFDAttentionModelRunnerV2,
)
from afd_plugin.validation import (
    assert_compatible_afd_stack,
    validate_gpu_model_runner_v2_config,
)


class AFDAttentionWorker(Worker):
    """Attention worker that constructs the AFD ModelRunnerV2 extension."""

    afd_expected_role = "attention"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker,
        )
        self.afd_runtime_deferred = is_elastic_attention(vllm_config)
        self.afd_kv_cache_config: KVCacheConfig | None = None
        if self.afd_runtime_deferred:
            self.elastic_ep_executor = AFDElasticGPUExecutor(self)

    # Native explicit KV budgets still run profile. Delay that forward until F
    # is receiving. Signature matches vLLM 0.26 Worker.determine_available_memory.
    def determine_available_memory(self) -> int:
        # ### PATCH START: defer AFD forward until cross-role rendezvous.
        if self.afd_runtime_deferred:
            budget = self.cache_config.kv_cache_memory_bytes
            assert budget is not None and budget > 0
            return budget
        # ### PATCH END: defer AFD forward until cross-role rendezvous.
        return super().determine_available_memory()

    # Preserve the complete upstream allocator/KV-zero/input initialization.
    # Signature matches vLLM 0.26 Worker.initialize_from_config.
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        # ### PATCH START: save config, never retain an old KV tensor allocation.
        if is_elastic_attention(self.vllm_config):
            self.afd_kv_cache_config = deepcopy(kv_cache_config)
        # ### PATCH END: save config, never retain an old KV tensor allocation.
        super().initialize_from_config(kv_cache_config)

    # Return the native contract during deferred EEP initialization; real warmup
    # follows after all target engines reach COMPLETE. Signature unchanged.
    def compile_or_warm_up_model(self) -> CompilationTimes:
        # ### PATCH START: EEP must not forward into stopped F workers.
        if self.afd_runtime_deferred:
            return CompilationTimes(language_model=0.0, encoder=0.0)
        # ### PATCH END: EEP must not forward into stopped F workers.
        return super().compile_or_warm_up_model()

    def afd_send_stop(self) -> None:
        send_stop(self)

    def afd_release_link(self) -> None:
        release_link(self)

    def afd_update_topology(self, afd: dict[str, str | int | bool]) -> None:
        update_topology(self, afd)

    def afd_connect(self) -> None:
        connect(self)

    def afd_finish(self) -> None:
        finish_attention(self)

    # Patch reason: vLLM 0.30.0 constructs its runner inside Worker.init_device
    # after an internal module import, with no injectable runner factory.
    # Patch functionality: validate the supported V2 contract, select the AFD model
    # identity before construction, and scope the module-class substitution to
    # the delegated native initialization window.
    # Signature: matches vLLM v0.30.0 Worker.init_device exactly: (self).
    # Upstream source: vllm/v1/worker/gpu_worker.py, Worker.init_device;
    # runner local-import/construct seam; commit
    # ced6857afa0ea7b2e3f0846a62e1394e90f15607. The 0.30.0 init_device still
    # local-imports GPUModelRunner from vllm.v1.worker.gpu.model_runner before
    # construction, so the module-attribute substitution stays effective.
    # Delegation exception: the large native device/distributed setup remains
    # in super().init_device(); only this missing runner-factory seam is local.
    # Removal/upstream plan: delete this substitution when vLLM adds runner
    # factory/class injection; the upstream target is that factory seam.
    # Concurrency invariant: Worker.init_device runs synchronously during startup,
    # with no concurrent runner construction in the worker process.
    def init_device(self):
        """Initialize the native GPU worker with the selected AFD runner."""

        # ### PATCH START: select and directly construct the AFD runner.
        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDAttentionWorker.init_device",
            expected_role="attention",
        )
        validate_gpu_model_runner_v2_config(
            self.vllm_config,
            expected_role="attention",
            device_type=self.device_config.device_type,
        )

        afd_model_config = get_afd_model_config(
            self.vllm_config.model_config,
            device_type="cuda",
        )
        self.vllm_config.model_config = afd_model_config
        self.model_config = afd_model_config

        native_runner_cls = gpu_model_runner_v2.GPUModelRunner
        gpu_model_runner_v2.GPUModelRunner = AFDAttentionModelRunnerV2
        try:
            super().init_device()
        finally:
            gpu_model_runner_v2.GPUModelRunner = native_runner_cls
        # ### PATCH END: select and directly construct the AFD runner.

        torch.accelerator.empty_cache()


__all__ = ["AFDAttentionWorker"]
