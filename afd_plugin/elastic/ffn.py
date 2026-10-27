# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A separate Ray FFN pool; no EngineCore, scheduler, or A routing identities."""

from __future__ import annotations

import asyncio
import copy
import os
from typing import TYPE_CHECKING

from afd_plugin.config import AFDConfig
from afd_plugin.elastic.config import ElasticTopology, set_afd_config

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from ray.util.placement_group import PlacementGroup
    from vllm.config import VllmConfig
    from vllm.v1.executor import Executor


class FFNActor:
    """Short control RPCs; model computation runs in the worker's existing thread.

    Runtime imports deliberately follow Ray device assignment in the actor,
    so importing this module on the API process does not initialize CUDA.
    """

    def __init__(self) -> None:
        import ray

        self.device_ids = ray.get_runtime_context().get_accelerator_ids()["GPU"]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.device_ids)
        os.environ["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"] = "0"
        self.executor: Executor | None = None

    def rendezvous(self) -> tuple[str, list[int]]:
        from vllm.utils.network_utils import get_ip, get_open_ports_list

        # Both AFD world rank 0 and F role rank 0 reside here. Allocate on the
        # host that actually binds the stores, not on the API host.
        return get_ip(), get_open_ports_list(2)

    def initialize_role(self, config: VllmConfig) -> None:
        from vllm.platforms.interface import set_assigned_physical_gpu_ids
        from vllm.v1.executor import Executor

        assert self.executor is None
        physical_ids = [int(device) for device in self.device_ids]
        config.parallel_config.assigned_physical_gpu_ids = physical_ids
        set_assigned_physical_gpu_ids(physical_ids)
        self.executor = Executor.get_class(config)(config)

    def collective_rpc(self, method: str, args: tuple = ()):
        assert self.executor is not None
        return self.executor.collective_rpc(method, args=args)

    def release_role(self) -> None:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None
            cleanup_dist_env_and_memory()


class FFNActorManager:
    def __init__(self, startup_config: VllmConfig):
        # API-side startup config still has the unaliased model identity.
        self.startup_config = copy.deepcopy(startup_config)
        self.actors: list[ActorHandle] = []
        self.placement_groups: list[PlacementGroup] = []

    async def resize_actors(self, size: int) -> None:
        import ray
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        from vllm.ray.ray_env import get_env_vars_to_copy

        # Existing roles have already been released collectively before removal.
        while len(self.actors) > size:
            ray.kill(self.actors.pop())
            ray.util.remove_placement_group(self.placement_groups.pop())

        env_names = get_env_vars_to_copy(destination="AFD FFNActor")
        env = {name: os.environ[name] for name in env_names if name in os.environ}
        env["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"] = "0"
        while len(self.actors) < size:
            pg = ray.util.placement_group(
                [{"GPU": 1, "CPU": 1}], strategy="STRICT_PACK"
            )
            self.placement_groups.append(pg)
            await pg.ready()
            actor = (
                ray.remote(FFNActor)
                .options(
                    num_gpus=1,
                    num_cpus=1,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=0
                    ),
                    runtime_env={"env_vars": env},
                )
                .remote()
            )
            self.actors.append(actor)

    async def rendezvous(self) -> tuple[str, list[int]]:
        return await self.actors[0].rendezvous.remote()

    async def initialize_roles(
        self, topology: ElasticTopology, afd: AFDConfig, role_port: int
    ) -> None:
        configs = []
        for rank in range(topology.ffn_dp):
            config = copy.deepcopy(self.startup_config)
            set_afd_config(config, afd)
            parallel = config.parallel_config
            parallel.enable_elastic_ep = False
            parallel.enable_eplb = False
            parallel.data_parallel_size = topology.ffn_dp
            parallel.data_parallel_size_local = 1
            parallel.data_parallel_rank = parallel.data_parallel_index = rank
            parallel.data_parallel_rank_local = 0
            parallel.tensor_parallel_size = topology.ffn_tp
            parallel.world_size = topology.ffn_tp
            parallel.data_parallel_master_ip = afd.host
            parallel.data_parallel_master_port = role_port
            parallel._data_parallel_master_port_list = [role_port]
            parallel._coord_store_port = None
            parallel.placement_group = None
            parallel.assigned_physical_gpu_ids = None
            parallel.worker_cls = "afd_plugin.v1.worker.AFDFFNWorker"
            parallel.distributed_executor_backend = "uni"
            configs.append(config)
        # Submit every rank before awaiting any collective initialization.
        await asyncio.gather(
            *(
                actor.initialize_role.remote(cfg)
                for actor, cfg in zip(self.actors, configs, strict=True)
            )
        )

    async def collective_rpc(self, method: str, args: tuple = ()) -> None:
        await asyncio.gather(
            *(actor.collective_rpc.remote(method, args) for actor in self.actors)
        )

    async def release_roles(self) -> None:
        await asyncio.gather(*(actor.release_role.remote() for actor in self.actors))

    def shutdown(self) -> None:
        """Force-stop owned actors on process teardown, including blocked recv.

        Normal resizes use STOP/join and release_roles. Shutdown may follow an
        A-side crash, so it must not rely on another A control message arriving.
        """
        import ray

        for actor in self.actors:
            ray.kill(actor)
        self.actors.clear()
        for pg in self.placement_groups:
            ray.util.remove_placement_group(pg)
        self.placement_groups.clear()
