# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import asyncio
import copy
import sys
import types
from types import SimpleNamespace

from afd_plugin.config import parse_afd_config
from afd_plugin.elastic.config import ElasticTopology
from afd_plugin.elastic.ffn import FFNActorManager


def test_shutdown_does_not_import_ray_during_interpreter_finalization(
    config, monkeypatch
):
    # The client registers a weakref finalizer on FFNActorManager, so shutdown
    # can run while the interpreter is tearing itself down (observed as
    # AsyncLLM.__del__ -> shutdown -> ImportError: sys.meta_path is None).
    # Nothing can be reaped at that point, and failing there hides the real
    # exit reason, so the guard must hold before the ray import.
    monkeypatch.setattr(sys, "is_finalizing", lambda: True, raising=False)
    monkeypatch.setitem(sys.modules, "ray", None)
    manager = FFNActorManager(config)
    actor = object()
    manager.actors = [actor]

    manager.shutdown()

    assert manager.actors == [actor], "finalizing must not touch the actor handles"


def test_ffn_configs_are_independent_and_initialize_collectively(config, monkeypatch):
    # initialize_roles picks the FFN worker class from the running platform, so
    # the seam needs a platform double, not a vLLM install: tests/unit must also
    # run on a host that has neither vLLM nor torch.
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(  # type: ignore[attr-defined]
        device_type="cpu"
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    original = copy.deepcopy(config)
    manager = FFNActorManager(config)
    seen = []

    async def initialize(cfg):
        seen.append(cfg)
        # Collective setup must enter all ranks before any awaits completion.
        await asyncio.sleep(0)
        assert len(seen) == 2

    manager.actors = [
        SimpleNamespace(initialize_role=SimpleNamespace(remote=initialize))
        for _ in range(2)
    ]
    topology = ElasticTopology(4, 2)
    afd = topology.afd_config(parse_afd_config(config), port=32100, role="ffn")
    asyncio.run(manager.initialize_roles(topology, afd, 32101))
    assert config == original
    for rank, cfg in enumerate(seen):
        parallel = cfg.parallel_config
        assert parallel.data_parallel_size == 2
        assert parallel.data_parallel_rank == rank
        assert parallel.data_parallel_rank_local == 0
        assert not parallel.enable_elastic_ep
        assert parallel.placement_group is None
        assert parallel.assigned_physical_gpu_ids is None
        assert parallel._data_parallel_master_port_list == [32101]
        assert cfg.additional_config["afd"]["role"] == "ffn"
    seen[0].parallel_config._data_parallel_master_port_list.pop()
    assert seen[1].parallel_config._data_parallel_master_port_list == [32101]
