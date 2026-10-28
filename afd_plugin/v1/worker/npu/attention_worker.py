# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU Attention-side worker for AFD execution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.worker_base import CompilationTimes
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.compat.npu import (
    apply_afd_ascend_patches_if_needed,
    fail_if_unsupported_npu_afd_features,
    fix_all2all_backend_for_afd,
    npu_afd_num_ubatches,
)
from afd_plugin.compat.npu.runtime_config import configure_async_cam_dbo
from afd_plugin.elastic.config import is_elastic_attention
from afd_plugin.elastic.npu import (
    AFDElasticNPUExecutor,
    connect,
    finish_attention,
    release_link,
    send_stop,
    update_topology,
)
from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
    AFDNPUAttentionModelRunnerV2,
)
from afd_plugin.validation import (
    NPU_ATTENTION_WORKER_FQCN,
    assert_compatible_afd_stack,
    validate_npu_model_runner_v2_config,
)


class AFDNPUAttentionWorker(NPUWorker):
    """Attention worker that creates an AFD-aware vLLM-Ascend runner."""

    afd_expected_role = "attention"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        apply_afd_ascend_patches_if_needed()
        super().__init__(*args, **kwargs)
        # NPUWorker.__init__ mounts the hardware-neutral
        # ElasticEPScalingExecutor; the elastic AFD path replaces it with the
        # NPU action executor (mirrors the GPU worker).
        self.afd_runtime_deferred = is_elastic_attention(self.vllm_config)
        self.afd_kv_cache_config: KVCacheConfig | None = None
        if self.afd_runtime_deferred:
            self.elastic_ep_executor = AFDElasticNPUExecutor(self)

    def init_device(self) -> None:
        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDNPUAttentionWorker.init_device",
            expected_role="attention",
            expected_worker_qualname_override=NPU_ATTENTION_WORKER_FQCN,
        )
        configure_async_cam_dbo(self.vllm_config)
        fail_if_unsupported_npu_afd_features(self.vllm_config)
        fix_all2all_backend_for_afd(self.vllm_config)
        validate_npu_model_runner_v2_config(
            self.vllm_config,
            expected_role="attention",
            device_type="npu",
        )

        self.device = self._init_device()
        init_workspace_manager(
            self.device,
            npu_afd_num_ubatches(self.vllm_config),
        )
        self.vllm_config.model_config = get_afd_model_config(
            self.vllm_config.model_config,
            device_type="npu",
        )
        self.model_runner = AFDNPUAttentionModelRunnerV2(
            self.vllm_config,
            self.device,
        )

    # Native explicit KV budgets still run profile. Delay that forward until F
    # is receiving. Signature matches vLLM-Ascend NPUWorker.determine_available_memory.
    def determine_available_memory(self) -> int:
        # ### PATCH START: defer AFD forward until cross-role rendezvous.
        if self.afd_runtime_deferred:
            budget = self.cache_config.kv_cache_memory_bytes
            assert budget is not None and budget > 0
            return budget
        # ### PATCH END: defer AFD forward until cross-role rendezvous.
        return super().determine_available_memory()

    # Preserve the complete upstream allocator/KV-zero/input initialization.
    # Signature matches vLLM-Ascend NPUWorker.initialize_from_config.
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


__all__ = ["AFDNPUAttentionWorker"]
