# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD pre/post steps around the native EEP A client, and independent F resize."""

from __future__ import annotations

import asyncio
import weakref
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

from afd_plugin.config import parse_afd_config
from afd_plugin.elastic.config import Role, set_afd_config, validate_elastic_config
from afd_plugin.elastic.ffn import FFNActorManager

if TYPE_CHECKING:
    from vllm.v1.engine.core_client import DPLBAsyncMPClient


class ElasticAFDClient:
    """Own only F and the cross-role lifecycle; native EEP still owns every A."""

    def __init__(self, client: DPLBAsyncMPClient):
        self.client = client
        self.topology = validate_elastic_config(client.vllm_config)
        self.ffn = FFNActorManager(client.vllm_config)
        self._finalizer = weakref.finalize(client, self.ffn.shutdown)

    async def initialize(self) -> None:
        await self.ffn.resize_actors(self.topology.ffn_dp)
        host, (link_port, role_port) = await self.ffn.rendezvous()
        base = replace(parse_afd_config(self.client.vllm_config), host=host)
        attention = self.topology.afd_config(base, port=link_port, role="attention")
        ffn = replace(attention, role="ffn")
        set_afd_config(self.client.vllm_config, attention)
        await asyncio.gather(
            self.client.collective_rpc_async(
                "afd_update_topology", args=(asdict(attention),)
            ),
            self.ffn.initialize_roles(self.topology, ffn, role_port),
        )
        await self.finish()

    async def prepare(self) -> None:
        # The outer AsyncLLM gate has drained requests, but only native DP pause
        # establishes consensus and prevents a late START_DP_WAVE/dummy forward.
        await self.client.pause_scheduler_async(mode="keep", clear_cache=True)
        await self.client.collective_rpc_async("afd_send_stop")
        await self.ffn.collective_rpc("join_ffn_server_loop")
        await asyncio.gather(
            self.client.collective_rpc_async("afd_release_link"),
            self.ffn.collective_rpc("afd_release_link"),
        )

    async def resize(self, size: int, role: Role) -> None:
        target = self.topology.resize(role, size)
        if target == self.topology:
            return
        await self.prepare()
        host, (link_port, role_port) = await self.ffn.rendezvous()
        base = replace(parse_afd_config(self.client.vllm_config), host=host)
        attention = target.afd_config(base, port=link_port, role="attention")
        ffn = replace(attention, role="ffn")
        # The API config is the source copied by the *native* A ActorManager.
        set_afd_config(self.client.vllm_config, attention)
        await self.client.collective_rpc_async(
            "afd_update_topology", args=(asdict(attention),)
        )
        if role == "attention":
            await self.ffn.collective_rpc("afd_update_topology", args=(asdict(ffn),))
            old_size = self.topology.attention_dp
            if size > old_size:
                await self.client._scale_up_elastic_ep(old_size, size)
            else:
                await self.client._scale_down_elastic_ep(old_size, size)
            # RECONFIGURE_FINISHED precedes COMPLETE. Query EngineCore itself,
            # yielding so its busy loop can continue state.progress().
            while not all(
                await asyncio.gather(
                    *(
                        self.client._call_utility_async(
                            "afd_eep_complete", engine=engine
                        )
                        for engine in self.client.core_engines
                    )
                )
            ):
                await asyncio.sleep(0.01)
        else:
            await self.ffn.release_roles()
            await self.ffn.resize_actors(size)
            await self.ffn.initialize_roles(target, ffn, role_port)
        await self.finish()
        self.topology = target

    async def finish(self) -> None:
        # Initiate both sides concurrently; connector construction is collective.
        await asyncio.gather(
            self.client.collective_rpc_async("afd_connect"),
            self.ffn.collective_rpc("afd_connect"),
        )
        await self.ffn.collective_rpc("start_ffn_server_loop")
        await self.client.collective_rpc_async("afd_finish")
        await self.client.call_utility_async("afd_reset_after_warmup")
        await self.client.resume_scheduler_async()
