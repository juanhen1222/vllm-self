# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy tests for HF3FS connector with dual instance cache sharing."""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import unittest

import lm_eval
import openai
import requests

from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_metadata_server import (
    Hf3fsMetadataServer,
)
from vllm.utils import get_open_port

NUM_CONCURRENT = 10
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03

SIMPLE_PROMPT = (
    "The best part about working on vLLM is that I got to meet so many people "
    "across various different organizations like UCB, Google, and Meta which means"
)

# MODEL_NAME = os.environ.get("TEST_MODEL", "/home/t4/models/Qwen3-32B")
MODEL_NAME = os.environ.get("TEST_MODEL", "/vllm-workspace/mnt/Qwen-8B/")
# MODEL_NAME = os.environ.get("TEST_MODEL", "/vllm-workspace/mnt/DeepSeek-Coder-V2-Lite/")


def run_simple_prompt(base_url: str = "http://localhost:8000/v1") -> None:
    """Run a simple prompt test with the given base URL."""
    client = openai.OpenAI(api_key="EMPTY", base_url=base_url)
    completion = client.completions.create(
        model=MODEL_NAME, prompt=SIMPLE_PROMPT, max_tokens=50, temperature=0.0
    )


class TestHf3fsConnectorAccuracy(unittest.TestCase):
    """Accuracy tests for HF3FS connector with dual instance cache sharing."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.storage_path = "/vllm-workspace/mnt/hf3fs"
        # self.storage_path = "/data/"
        os.makedirs(self.storage_path, exist_ok=True)

        self._setup_server_ports()
        self._start_metadata_server()
        self._configure_kv_settings()

        self.vllm_proc_1 = None
        self.vllm_proc_2 = None

    def _setup_server_ports(self) -> None:
        """Set up server ports and URLs."""
        self.vllm_port_1 = get_open_port()
        self.vllm_port_2 = get_open_port()

        self.base_url_v1_1 = f"http://localhost:{self.vllm_port_1}/v1"
        self.base_url_1 = f"http://localhost:{self.vllm_port_1}"

        self.base_url_v1_2 = f"http://localhost:{self.vllm_port_2}/v1"
        self.base_url_2 = f"http://localhost:{self.vllm_port_2}"

    def _start_metadata_server(self) -> None:
        """Start metadata server in background thread."""
        print("Starting metadata server...")
        self.metadata_server = Hf3fsMetadataServer(
            persistence_path=None, save_interval=3600
        )

        self.server_thread = threading.Thread(
            target=self._run_metadata_server, daemon=True
        )
        self.server_thread.start()
        time.sleep(1)
        print("Metadata server started")

    def _configure_kv_settings(self) -> None:
        """Configure KV connector settings."""
        self.base_kv_config = {
            "kv_connector": "hf3fs",
            "kv_connector_extra_config": {
                "shared_storage_path": self.storage_path,
                "hf3fs_storage_path": self.storage_path,
                "hf3fs_file_size": 1024 * 1024 * 1024 * 10,
                "hf3fs_metadata_server_url": "http://localhost:18000",
                "hf3fs_client_numjobs": 4,
            },
        }

        self.kv_config_producer = {
            **self.base_kv_config,
            "kv_role": "kv_both",
        }

        self.kv_config_consumer = {
            **self.base_kv_config,
            "kv_role": "kv_consumer",
        }

    def _run_metadata_server(self) -> None:
        """Run metadata server in background thread."""
        try:
            import uvicorn

            uvicorn.run(
                self.metadata_server.app,
                host="127.0.0.1",
                port=18000,
                log_level="warning",
            )
        except Exception as e:
            print(f"Metadata server error: {e}")

    def _start_vllm_server(self, server_id: int, kv_config: dict, port: int) -> None:
        """Start VLLM server with HF3FS connector configuration."""
        server_name = "Producer" if server_id == 1 else "Consumer"
        print(f"Starting VLLM {server_name} server on port {port}...")

        vllm_args = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--port",
            str(port),
            "--trust-remote-code",
            "--enforce-eager",
            "--tensor-parallel-size",
            "2",
            "--max-model-len",
            "2048",
            "--kv-transfer-config",
            json.dumps(kv_config),
            "--gpu-memory-utilization",
            "0.8",
            "--max-num-batched-tokens",
            "512",
        ]

        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if server_id == 2:
            env["CUDA_VISIBLE_DEVICES"] = "2,3"

        proc = subprocess.Popen(
            vllm_args,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        if server_id == 1:
            self.vllm_proc_1 = proc
        else:
            self.vllm_proc_2 = proc

        self._wait_for_vllm_server(server_id, port)
        print(f"VLLM {server_name} server started successfully")

    def _wait_for_vllm_server(
        self, server_id: int, port: int, timeout: int = 240
    ) -> None:
        """Wait for VLLM server to be ready."""
        health_url = f"http://localhost:{port}/health"
        start_time = time.time()
        server_name = "Producer" if server_id == 1 else "Consumer"
        proc = self.vllm_proc_1 if server_id == 1 else self.vllm_proc_2

        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    return
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError(f"VLLM {server_name} server exited unexpectedly")
                time.sleep(2)

        raise RuntimeError(
            f"VLLM {server_name} server failed to start within {timeout} seconds"
        )

    def _stop_vllm_server(self, server_id: int = None) -> None:
        """Stop VLLM server(s)."""
        if server_id is None:
            self._stop_vllm_server(1)
            self._stop_vllm_server(2)
            return

        server_name = "Producer" if server_id == 1 else "Consumer"
        proc = self.vllm_proc_1 if server_id == 1 else self.vllm_proc_2

        if proc is not None:
            print(f"Stopping VLLM {server_name} server...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"Force killing VLLM {server_name} server...")
                proc.kill()

            if server_id == 1:
                self.vllm_proc_1 = None
            else:
                self.vllm_proc_2 = None

    def tearDown(self) -> None:
        """Clean up test fixtures and stop servers."""
        self._stop_vllm_server()
        shutil.rmtree(self.storage_path, ignore_errors=True)
        print("Test cleanup completed")

    def test_dual_instance_cache_sharing(self) -> None:
        """Test cache sharing between two VLLM instances via HF3FS."""
        producer_results = None
        consumer_results = None

        try:
            print("=" * 60)
            print("PHASE 1: CACHE PRODUCER - Generating cache to HF3FS")
            print("=" * 60)

            self._start_vllm_server(1, self.kv_config_producer, self.vllm_port_1)
            self._test_basic_functionality(1)

            print("Running evaluation on Producer server to generate cache...")
            producer_results = self._run_lm_eval_accuracy(1)

            print("Stopping Producer server...")
            self._stop_vllm_server(1)
            time.sleep(3)

            print("=" * 60)
            print("PHASE 2: CACHE CONSUMER - Using existing cache from HF3FS")
            print("=" * 60)

            self._start_vllm_server(2, self.kv_config_consumer, self.vllm_port_2)

            print("Running evaluation on Consumer server using cached data...")
            consumer_results = self._run_lm_eval_accuracy(2)

            self._compare_results(producer_results, consumer_results)

        except Exception as e:
            print(f"Dual instance test failed with error: {e}")
            raise
        finally:
            self._stop_vllm_server()

    def test_single_instance_baseline(self) -> None:
        """Test single instance without cache sharing for baseline comparison."""
        try:
            print("=" * 60)
            print("PHASE 1: CACHE - Generating cache to HF3FS")
            print("=" * 60)

            self._start_vllm_server(1, self.kv_config_producer, self.vllm_port_1)
            self._test_basic_functionality(1)

            print("Running evaluation on server to generate cache...")
            results = self._run_lm_eval_accuracy(1)

            requests.post(f"{self.base_url_1}/reset_prefix_cache")

            print("Running evaluation on server again...")
            results1 = self._run_lm_eval_accuracy(1)

            self._compare_results(results, results1)

        except Exception as e:
            print(f"Baseline test failed with error: {e}")
            raise
        finally:
            self._stop_vllm_server(1)

    def _test_basic_functionality(self, server_id: int) -> None:
        """Test basic server functionality before running accuracy test."""
        server_name = "Producer" if server_id == 1 else "Consumer"
        base_url = self.base_url_v1_1 if server_id == 1 else self.base_url_v1_2

        print(f"Testing basic {server_name} server functionality...")

        client = openai.OpenAI(api_key="EMPTY", base_url=base_url)

        completion = client.completions.create(
            model=MODEL_NAME, prompt=SIMPLE_PROMPT, max_tokens=50, temperature=0.0
        )

        assert (
            completion.choices[0].text is not None
        ), f"{server_name} server did not return valid completion"
        print(
            f"{server_name} basic functionality test passed. Response: {completion.choices[0].text[:100]}..."
        )

    def _run_lm_eval_accuracy(self, server_id: int) -> dict:
        """Run lm_eval accuracy test on specified server."""
        server_name = "Producer" if server_id == 1 else "Consumer"
        base_url = self.base_url_v1_1 if server_id == 1 else self.base_url_v1_2

        print(
            f"Running lm_eval accuracy test on {server_name} server for {MODEL_NAME}..."
        )
        start_time = time.time()

        model_args = (
            f"model={MODEL_NAME},"
            f"base_url={base_url}/completions,"
            f"num_concurrent={NUM_CONCURRENT},"
            f"tokenized_requests=False"
        )

        results = lm_eval.simple_evaluate(
            model="local-completions",
            model_args=model_args,
            tasks=TASK,
            limit=10,
        )

        evaluation_time = time.time() - start_time
        measured_value = results["results"][TASK][FILTER]
        expected_value = 0.5

        print(f"{server_name} - Measured accuracy: {measured_value:.4f}")
        print(f"{server_name} - Expected accuracy: {expected_value:.4f} ± {RTOL:.3f}")
        print(f"{server_name} - Evaluation time: {evaluation_time:.2f} seconds")

        result_summary = {
            "server_name": server_name,
            "server_id": server_id,
            "accuracy": measured_value,
            "evaluation_time": evaluation_time,
            "expected_accuracy": expected_value,
            "full_results": results,
        }

        assert measured_value >= expected_value - RTOL, (
            f"{server_name} accuracy {measured_value:.4f} is too low "
            f"(expected >= {expected_value - RTOL:.4f})"
        )

        return result_summary

    def _compare_results(self, producer_results: dict, consumer_results: dict) -> None:
        """Compare results between producer and consumer instances."""
        print("\n" + "=" * 60)
        print("RESULTS COMPARISON")
        print("=" * 60)

        print(f"Producer accuracy: {producer_results['accuracy']:.4f}")
        print(f"Consumer accuracy: {consumer_results['accuracy']:.4f}")

        accuracy_diff = abs(producer_results["accuracy"] - consumer_results["accuracy"])
        print(f"Accuracy difference: {accuracy_diff:.4f}")

        print(f"\nProducer evaluation time: {producer_results['evaluation_time']:.2f}s")
        print(f"Consumer evaluation time: {consumer_results['evaluation_time']:.2f}s")

        if consumer_results["evaluation_time"] < producer_results["evaluation_time"]:
            speedup = (
                producer_results["evaluation_time"]
                / consumer_results["evaluation_time"]
            )
            print(f"Consumer speedup: {speedup:.2f}x faster")
        else:
            slowdown = (
                consumer_results["evaluation_time"]
                / producer_results["evaluation_time"]
            )
            print(f"Consumer slowdown: {slowdown:.2f}x slower")

        max_allowed_diff = 0.05

        assert accuracy_diff <= max_allowed_diff, (
            f"Accuracy difference too large: {accuracy_diff:.4f} > {max_allowed_diff:.4f}. "
            f"Producer: {producer_results['accuracy']:.4f}, "
            f"Consumer: {consumer_results['accuracy']:.4f}"
        )

        print(
            f"✓ Accuracy difference {accuracy_diff:.4f} <= {max_allowed_diff:.4f} (acceptable)"
        )

        if consumer_results["evaluation_time"] < producer_results["evaluation_time"]:
            print("✓ Cache provided performance benefit")
        else:
            print(
                "⚠ Cache did not provide performance benefit (may be due to small test size)"
            )


if __name__ == "__main__":
    unittest.main()
