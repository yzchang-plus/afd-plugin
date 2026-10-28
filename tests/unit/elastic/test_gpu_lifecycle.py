"""Device operations are doubles; worker lifecycle/order is production code."""

import ast
import copy
import importlib.util
import sys
import types
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from afd_plugin.config import parse_afd_config

ROOT = Path(__file__).parents[3]


@pytest.fixture
def gpu(monkeypatch):
    events = []
    workspace = [False]

    def init_workspace(device, num_ubatches):
        workspace[0] = num_ubatches
        events.append("workspace_init")

    def unlock():
        assert workspace[0], "workspace reset during runner shutdown"
        events.append("unlock")

    class Base:
        def __init__(self, worker):
            self.worker = worker

    def create_groups(**kwargs):
        assert not kwargs["enable_eplb"]
        events.append(("create_groups", kwargs["new_dp_size"]))

    stubs = {
        "torch": {
            "inference_mode": lambda: lambda fn: fn,
            "cuda": SimpleNamespace(
                synchronize=lambda *args: events.append("sync"),
                empty_cache=lambda: events.append("empty_cache"),
            ),
            "compiler": SimpleNamespace(reset=lambda: events.append("compile_reset")),
        },
        "vllm.compilation.cuda_graph": {
            "CUDAGraphWrapper": SimpleNamespace(
                clear_all_graphs=lambda: events.append("clear_graphs")
            )
        },
        "vllm.compilation.wrapper": {"reset_compile_wrapper": lambda model: None},
        "vllm.config": {"set_current_vllm_config": lambda config: nullcontext()},
        "vllm.distributed": {"get_world_group": lambda: None},
        "vllm.distributed.elastic_ep.elastic_execute": {
            "ElasticEPScalingExecutor": Base
        },
        "vllm.distributed.elastic_ep.standby_state": {
            "create_standby_groups": create_groups,
            "pop_standby_groups": lambda: {},
        },
        "vllm.distributed.parallel_state": {
            "_replace_active_groups": lambda **kwargs: events.append("switch_groups")
        },
        "vllm.v1.engine": {
            "ReconfigureDistributedRequest": SimpleNamespace,
            "ReconfigureRankType": SimpleNamespace(KEEP_CURRENT_RANK=-1),
        },
        "vllm.v1.worker.workspace": {
            "init_workspace_manager": init_workspace,
            "unlock_workspace": unlock,
            "lock_workspace": lambda: events.append("lock"),
        },
        "afd_plugin.connectors": {
            "AFDConnectorFactory": None,
            "AFDControlPayload": lambda *args, **kwargs: SimpleNamespace(**kwargs),
        },
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "_afd_gpu_cpu_test", ROOT / "afd_plugin/elastic/gpu.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, events=events, workspace=workspace)


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_topology_rpc_restores_config_after_msgpack(gpu, config, role):
    msgspec = pytest.importorskip("msgspec")
    previous = parse_afd_config(config)
    target = replace(
        previous,
        role=role,
        host="10.0.0.2",
        port=30000,
        num_attention_ranks=4,
        num_ffn_ranks=2,
    )
    # EngineCore collective_rpc has an untyped nested args tuple. Msgspec
    # decodes dataclasses/maps there as dicts, without restoring worker types.
    (decoded,) = msgspec.msgpack.decode(msgspec.msgpack.encode((asdict(target),)))
    wrapper = SimpleNamespace(afd_config=previous)
    runner = SimpleNamespace(
        afd_config=previous,
        get_model=lambda: SimpleNamespace(modules=lambda: [wrapper]),
    )
    worker = SimpleNamespace(
        vllm_config=config,
        model_runner=runner,
        afd_expected_role=role,
    )
    gpu.module.update_topology(worker, decoded)
    assert runner.afd_config == target
    assert wrapper.afd_config == target
    assert parse_afd_config(config) == target


@pytest.mark.parametrize("enable_dbo", [False, True])
def test_reload_recreates_workspace_before_model_and_full_kv(
    gpu, monkeypatch, config, enable_dbo
):
    events = gpu.events
    config.parallel_config.enable_dbo = enable_dbo
    config.parallel_config.use_ubatching = enable_dbo

    def old_shutdown():
        events.append("shutdown")
        gpu.workspace[0] = False

    class Runner:
        def __init__(self, cfg, device):
            assert gpu.workspace[0] == (2 if enable_dbo else 1)
            events.append("runner")

    module = types.ModuleType("afd_plugin.v1.worker.attention_model_runner")
    module.AFDAttentionModelRunner = Runner
    monkeypatch.setitem(sys.modules, module.__name__, module)
    saved_kv = {"groups": ["attention"], "blocks": 16}

    def initialize_kv(kv):
        assert kv == saved_kv and kv is not saved_kv
        events.append("full_kv_init")

    worker = SimpleNamespace(
        model_runner=SimpleNamespace(shutdown=old_shutdown),
        device="cuda:0",
        parallel_config=config.parallel_config,
        vllm_config=config,
        afd_kv_cache_config=saved_kv,
        load_model=lambda **kwargs: events.append(
            ("checkpoint", kwargs["load_dummy_weights"])
        ),
        initialize_from_config=initialize_kv,
    )
    executor = gpu.module.AFDElasticGPUExecutor(worker)
    executor.reconfig_request = SimpleNamespace(
        new_data_parallel_size=4,
        new_data_parallel_rank=-1,
        new_data_parallel_rank_local=-1,
        new_data_parallel_master_ip="10.0.0.1",
        new_data_parallel_master_port=30000,
        new_data_parallel_master_port_list=[30001],
        coord_store_port=30002,
    )
    executor.switch_and_prepare()
    assert (
        events.index("shutdown")
        < events.index("workspace_init")
        < events.index("runner")
    )
    assert (
        events.index("switch_groups")
        < events.index(("checkpoint", False))
        < events.index("full_kv_init")
    )
    assert worker.parallel_config.data_parallel_size == 4


def test_new_a_loads_checkpoint_without_receiving_experts(gpu):
    worker = SimpleNamespace(load_model=lambda **kwargs: gpu.events.append(kwargs))
    executor = gpu.module.AFDElasticGPUExecutor(worker)
    executor.load_model()
    executor.broadcast_expert_mapping()
    executor.transfer_weights(2, 4)
    executor.receive_weights()
    executor.prepare_new_worker()
    executor.perform_eplb_reshuffle()
    executor.perform_scale_down_eplb_reshuffle(2)
    executor.rewarm_workspace()
    assert gpu.events == [{"load_dummy_weights": False}]


def test_finish_releases_old_kv_before_allocating_and_enables_real_warmup(gpu):
    gpu.workspace[0] = True
    events = gpu.events
    worker = SimpleNamespace(
        afd_runtime_deferred=True, vllm_config=None, device="cuda:0"
    )

    def profile():
        assert not worker.afd_runtime_deferred
        events.append("profile")

    worker.model_runner = SimpleNamespace(
        _cleanup_profiling_kv_cache=lambda: events.append("free_kv"),
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(clear=lambda: events.append("clear_blocks"))
        ),
        profile_run=profile,
    )
    worker.afd_kv_cache_config = {"blocks": 16}
    worker.initialize_from_config = lambda cfg: events.append("allocate_kv")
    worker.compile_or_warm_up_model = lambda: events.append("warmup")
    gpu.module.finish_attention(worker)
    assert (
        events.index("free_kv")
        < events.index("allocate_kv")
        < events.index("profile")
        < events.index("warmup")
    )
    assert events[-2:] == ["clear_blocks", "sync"]


def _worker_methods(class_name, methods):
    """Compile actual method bodies without importing the native GPU worker."""
    path = ROOT / "afd_plugin/v1/worker/ffn_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    tree.body = [
        copy.deepcopy(node)
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    namespace = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                set_device=lambda device: None, synchronize=lambda *args: None
            )
        )
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def test_stop_exits_before_touching_batch_or_model():
    methods = _worker_methods("AFDFFNWorker", {"_run_ffn_server_loop"})
    # STOP has no metadata/graph fields. Any attempt to run a batch will fail.
    runner = SimpleNamespace(
        connector=SimpleNamespace(
            control_plane=SimpleNamespace(
                recv_dp_metadata_list=lambda: SimpleNamespace(stop=True)
            )
        )
    )
    worker = SimpleNamespace(
        _ffn_shutdown_event=SimpleNamespace(is_set=lambda: False),
        model_runner=runner,
        device=SimpleNamespace(type="cuda"),
    )
    methods["_run_ffn_server_loop"](worker)


def test_join_timeout_keeps_live_thread_handle():
    methods = _worker_methods("AFDFFNWorker", {"join_ffn_server_loop"})
    thread = SimpleNamespace(join=lambda **kwargs: None, is_alive=lambda: True)
    worker = SimpleNamespace(_ffn_thread=thread)
    with pytest.raises(TimeoutError, match="STOP"):
        methods["join_ffn_server_loop"](worker)
    assert worker._ffn_thread is thread
