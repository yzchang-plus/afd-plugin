# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Host-only harness integration: real sockets, processes and signals; no devices."""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.e2e import runner
from tests.e2e.models.deepseek_v4_flash import test_async_cam_npu as entrypoint

CONCURRENT_REQUESTS = 10
# Shared argv for the elastic scenario; parsed through the same argument and
# scenario-configuration path the E2E driver uses.
ELASTIC_ARGV = (
    "--model",
    "/models/deepseek-v2-lite",
    "--scenario",
    runner.ELASTIC_SCENARIO,
    "--device-backend",
    "npu",
    "--gsm8k-output-path",
    "/tmp/gsm8k-elastic",
)


def elastic_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    runner.add_scenario_arguments(parser)
    args = parser.parse_args(ELASTIC_ARGV)
    runner.configure_scenario(args)
    return args


def flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


CANCEL_SCRIPT = """
import json
import sys
from pathlib import Path
from tests.e2e import runner

pid_path = Path(sys.argv.pop(1))
pids = []
start = runner.start_process

def start_service(name, command, env):
    process = start(name, [sys.executable, '-c', 'import time; time.sleep(60)'], env)
    pids.append(process.pid)
    pid_path.write_text(json.dumps(pids))
    return process

runner.start_process = start_service
runner.wait_for_openai_api = lambda *args: None
sys.exit(runner.main())
"""


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_cancel_blocked_requests_cleans_service_groups(
    monkeypatch, tmp_path, signum
):  # long-test: real HTTP cancellation requires owned subprocess setup and teardown
    monkeypatch.setenv("AFD_E2E_BACKEND", "npu")
    monkeypatch.setenv("AFD_E2E_DEVICES", ",".join(map(str, range(16))))
    monkeypatch.setenv("AFD_NPU_E2E_MODEL", "/models/dsv4")
    monkeypatch.setenv("HCCL_IF_IP", "192.0.2.1")
    arrived = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    request_count = 0

    class StalledServer(ThreadingHTTPServer):
        request_queue_size = 32

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal request_count
            self.rfile.read(int(self.headers["Content-Length"]))
            with lock:
                request_count += 1
                if request_count == CONCURRENT_REQUESTS:
                    arrived.set()
            release.wait(timeout=30)

    server = StalledServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    pid_path = tmp_path / "services.json"
    command = entrypoint.build_runner_command(tmp_path / "responses.json")
    command.extend(["--api-port-base", str(server.server_port)])
    with (tmp_path / "cancel.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", CANCEL_SCRIPT, str(pid_path), *command[3:]],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            assert arrived.wait(timeout=15), "ten requests never reached the server"
            process.send_signal(signum)
            # Much shorter than both the HTTP timeout (300s) and the outer
            # runner cleanup deadline (240s). Exercise real signal unwinding,
            # socket cancellation and separately launched service groups.
            process.wait(timeout=10)
            log.seek(0)
            assert process.returncode == 128 + signum, log.read()
            for pid in json.loads(pid_path.read_text()):
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)
            results = json.loads((tmp_path / "responses.json").read_text())
            assert len(results) == CONCURRENT_REQUESTS
            assert all("cancelled" in row["error"] for row in results)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            if pid_path.exists():
                for pid in json.loads(pid_path.read_text()):
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(pid, signal.SIGKILL)
            release.set()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)


@pytest.mark.parametrize("device_count", [4, 5, 6, 7, 8, 16])
def test_plan_elastic_ladder_keeps_every_step_legal(device_count):
    attention_dp, ffn_dp = 2, 2
    for role, new_size in runner.plan_elastic_ladder(device_count):
        if role == "attention":
            attention_dp = new_size
        else:
            ffn_dp = new_size
        assert attention_dp >= ffn_dp
        assert attention_dp % ffn_dp == 0
        assert attention_dp <= runner.ELASTIC_MAX_DP_SIZE
        assert attention_dp + ffn_dp <= device_count
    assert (attention_dp, ffn_dp) == (2, 2)


@pytest.mark.parametrize(
    ("device_count", "expected_steps"),
    [
        (3, []),
        (4, [("ffn", 1), ("ffn", 2)]),
        (6, [("ffn", 1), ("ffn", 2), ("attention", 4), ("attention", 2)]),
        (
            8,
            [
                ("ffn", 1),
                ("ffn", 2),
                ("attention", 4),
                ("ffn", 4),
                ("ffn", 2),
                ("attention", 2),
            ],
        ),
    ],
)
def test_plan_elastic_ladder_matches_device_tiers(device_count, expected_steps):
    assert runner.plan_elastic_ladder(device_count) == expected_steps


def test_elastic_launch_command_carries_the_single_serve_shape():
    command = runner.build_vllm_command(elastic_args(), role="attention")
    assert flag_value(command, "--worker-cls") == runner.ELASTIC_ATTENTION_WORKER_CLS
    assert flag_value(command, "--data-parallel-backend") == "ray"
    assert flag_value(command, "--distributed-executor-backend") == "uni"
    assert "--enable-elastic-ep" in command
    assert flag_value(command, "--elastic-ep-max-dp-size") == str(
        runner.ELASTIC_MAX_DP_SIZE
    )
    assert flag_value(command, "--data-parallel-size") == "2"
    assert flag_value(command, "--data-parallel-size-local") == "2"
    assert flag_value(command, "--kv-cache-memory-bytes") == str(
        runner.ELASTIC_KV_CACHE_MEMORY_BYTES
    )
    # Ascend cannot retire ACL graphs mid-resize, so eager is part of the
    # contract, not a default the scenario happens to inherit.
    assert "--enforce-eager" in command
    assert "--max-cudagraph-capture-size" not in command
    additional_config = json.loads(flag_value(command, "--additional-config"))
    assert additional_config["afd"]["role"] == "attention"
    assert additional_config["afd"]["connector"] == "CAMP2pAFDConnector"
    assert additional_config["afd"]["num_attention_ranks"] == 2
    assert additional_config["afd"]["num_ffn_ranks"] == 2
    assert additional_config["afd_elastic"] == {
        "ffn_tensor_parallel_size": runner.ELASTIC_FFN_TENSOR_PARALLEL_SIZE,
    }


@pytest.mark.parametrize(
    ("device_backend", "attention_devices", "ffn_devices", "match"),
    [
        ("gpu", ["0", "1", "2", "3"], [], "NPU backend"),
        ("npu", ["0", "1", "2", "3"], ["4"], "ffn-devices"),
        ("npu", ["0", "1"], [], "at least 4 devices"),
        ("npu", ["0", "0", "1", "2"], [], "unique"),
    ],
)
def test_elastic_topology_rejects_invalid_launches(
    device_backend, attention_devices, ffn_devices, match
):
    args = elastic_args()
    args.device_backend = device_backend
    with pytest.raises(ValueError, match=match):
        runner.validate_topology(args, attention_devices, ffn_devices)


def test_elastic_topology_accepts_every_device_on_one_serve():
    runner.validate_topology(elastic_args(), ["0", "1", "2", "3"], [])


def test_elastic_deployment_launches_attention_only():
    role_devices, launch_order = runner.deployment_plan(
        elastic_args(),
        ["0", "1", "2", "3"],
        [],
    )
    assert role_devices == {"attention": ["0", "1", "2", "3"]}
    assert launch_order == (("attention", "ATTN"),)


def test_elastic_negative_resize_target_exceeds_the_max_dp_bound():
    # If the refused request ever fit under --elastic-ep-max-dp-size the
    # negative path would resize for real and stop proving the preflight.
    assert runner.ELASTIC_REJECTED_ATTENTION_DP > runner.ELASTIC_MAX_DP_SIZE


def test_elastic_scenario_stays_outside_the_gpu_v2_gate():
    # V2_SCENARIOS membership makes the runner require GPU, so the elastic
    # case must keep its own NPU-only validation.
    assert runner.ELASTIC_SCENARIO not in runner.V2_SCENARIOS
