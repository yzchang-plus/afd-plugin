import copy

import pytest

from afd_plugin.config import parse_afd_config
from afd_plugin.elastic.config import (
    ElasticTopology,
    is_elastic_attention_worker,
    set_afd_config,
    validate_elastic_config,
)


def test_default_and_independent_role_sizes(config):
    topology = validate_elastic_config(config)
    assert topology == ElasticTopology(2, 1)
    assert topology.resize("attention", 4) == ElasticTopology(4, 1)
    assert topology.resize("ffn", 2) == ElasticTopology(2, 2)
    # GPU's existing non-divisible topology is retained.
    assert ElasticTopology(3, 1).resize("ffn", 2) == ElasticTopology(3, 2)


@pytest.mark.parametrize("size", [0, -1, True, 1.5, "2"])
def test_bad_size_is_rejected(size):
    with pytest.raises(ValueError):
        ElasticTopology(4, 1).resize("ffn", size)


def test_unsupported_targets_fail_before_mutation():
    topology = ElasticTopology(2, 1)
    for role, size in [("attention", 1), ("ffn", 3), ("both", 2)]:
        with pytest.raises(ValueError):
            topology.resize(role, size)
    assert topology == ElasticTopology(2, 1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enable_eplb", True),
        ("enable_expert_parallel", False),
        ("data_parallel_backend", "mp"),
        ("distributed_executor_backend", "ray"),
        ("data_parallel_external_lb", True),
        ("data_parallel_hybrid_lb", True),
        ("pipeline_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("use_ubatching", True),
        ("_api_process_count", 2),
        ("worker_cls", "auto"),
    ],
)
def test_runtime_boundaries(config, field, value):
    setattr(config.parallel_config, field, value)
    with pytest.raises(ValueError):
        validate_elastic_config(config)


def test_npu_gate_is_explicit(config):
    config.additional_config["afd"]["connector"] = "CAMP2pAFDConnector"
    config.parallel_config.worker_cls = "afd_plugin.v1.worker.npu.AFDNPUAttentionWorker"
    with pytest.raises(ValueError, match="stateless HCCL"):
        validate_elastic_config(config)


def test_no_global_eplb_exemption(config):
    parallel = config.parallel_config
    assert is_elastic_attention_worker(parallel)
    parallel.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
    assert not is_elastic_attention_worker(parallel)
    parallel.worker_cls = "afd_plugin.v1.worker:AFDAttentionWorker"
    assert is_elastic_attention_worker(parallel)
    parallel.enable_elastic_ep = False
    assert not is_elastic_attention_worker(parallel)


def test_config_update_preserves_connector_options_and_input(config):
    config.additional_config["afd"]["connector_extra_config"] = {"test": 12}
    old = copy.deepcopy(config.additional_config)
    afd = ElasticTopology(4, 2).afd_config(
        parse_afd_config(config), port=12345, role="attention"
    )
    set_afd_config(config, afd)
    assert config.additional_config["afd"]["connector_extra_config"] == {"test": 12}
    assert config.additional_config["afd"]["num_ffn_ranks"] == 2
    assert old["afd"]["num_ffn_ranks"] == 1
