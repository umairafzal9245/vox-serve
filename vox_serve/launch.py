"""FastAPI server entry point and request orchestration for VoxServe."""

import asyncio
import atexit
import audioop
import base64
import collections
import io
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Dict, Optional

import torch
import zmq
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .utils import get_global_log_level, get_logger, set_global_log_level

# Module-level logger - will be updated with proper log level in main()
logger = get_logger(__name__)


class APIServer:
    """Manage scheduler processes and route API requests to the model backends."""
    def __init__(
        self,
        model_name: str = "canopylabs/orpheus-3b-0.1-ft",
        scheduler_type: str = "base",
        request_socket_path: str = "/tmp/vox_serve_request.ipc",
        result_socket_path: str = "/tmp/vox_serve_result.ipc",
        output_dir: str = "/tmp/vox_serve_audio",
        timeout_seconds: float = 600.0,
        max_batch_size: int = 8,
        top_p: float = None,
        top_k: int = None,
        min_p: float = None,
        temperature: float = None,
        max_tokens: int = None,
        repetition_penalty: float = None,
        repetition_window: int = None,
        cfg_scale: float = None,
        greedy: bool = False,
        enable_cuda_graph: bool = True,
        enable_disaggregation: bool = False,
        enable_nvtx: bool = False,
        enable_torch_compile: bool = False,
        unroll_depth_cuda_graph: bool = False,
        max_num_pages: int = None,
        page_size: int = 2048,
        async_scheduling: bool = False,
        dp_size: int = 1,
        detokenize_interval: int = None,
    ):
        """Initialize the API server and start scheduler process(es).

        Args:
            model_name: Model identifier or local path.
            scheduler_type: Scheduler backend to use.
            request_socket_path: IPC path for request socket (without rank suffix).
            result_socket_path: IPC path for result socket.
            output_dir: Directory to write generated audio and uploads.
            timeout_seconds: Per-request timeout in seconds.
            max_batch_size: Maximum batch size for scheduler inference.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            min_p: Min-p sampling parameter.
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.
            repetition_penalty: Repetition penalty value.
            repetition_window: Repetition window size.
            cfg_scale: Classifier-free guidance scale.
            greedy: Enable greedy decoding.
            enable_cuda_graph: Enable CUDA graph optimization.
            enable_disaggregation: Enable disaggregated execution (multi-GPU).
            enable_nvtx: Enable NVTX profiling.
            enable_torch_compile: Enable torch.compile optimization.
            max_num_pages: Maximum number of KV cache pages.
            page_size: Size of each KV cache page.
            async_scheduling: Enable async scheduling mode.
            dp_size: Data parallel replica count.
            detokenize_interval: Interval for audio detokenization (model-specific).
        """
        self.model_name = model_name
        self.request_socket_path = request_socket_path
        self.result_socket_path = result_socket_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.upload_dir = Path(output_dir) / "uploads"
        self.upload_dir.mkdir(exist_ok=True)
        # Persistent registry of pre-uploaded reference voices (register-once).
        # Lets clients upload a reference a single time and then reference it by
        # voice_id on subsequent /generate calls, avoiding per-request upload.
        self.voices_dir = Path(output_dir) / "voices"
        self.voices_dir.mkdir(exist_ok=True)
        self.voices: Dict[str, str] = {}  # voice_id -> stored audio path
        self.voices_lock = threading.Lock()
        # Reload voices persisted from previous runs (filename stem == voice_id)
        for _vf in self.voices_dir.glob("*"):
            if _vf.is_file():
                self.voices[_vf.stem] = str(_vf)
        self.timeout_seconds = timeout_seconds
        self.max_batch_size = max_batch_size
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.repetition_window = repetition_window
        self.cfg_scale = cfg_scale
        self.greedy = greedy
        self.enable_cuda_graph = enable_cuda_graph
        self.enable_disaggregation = enable_disaggregation
        self.enable_nvtx = enable_nvtx
        self.enable_torch_compile = enable_torch_compile
        self.unroll_depth_cuda_graph = unroll_depth_cuda_graph
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.scheduler_type = scheduler_type
        self.async_scheduling = async_scheduling
        self.dp_size = dp_size
        self.detokenize_interval = detokenize_interval
        self.scheduler_processes = None  # Will be a list for DP mode
        self.logger = get_logger(__name__)

        # Concurrent request tracking
        self.pending_requests: Dict[str, Dict] = {}  # request_id -> {chunks: [], event: threading.Event()}
        # Track recently completed request_ids to absorb late messages without warnings
        self.recently_completed = collections.OrderedDict()  # request_id -> timestamp
        self.recently_completed_ttl_sec = 5.0
        self.request_lock = threading.Lock()
        self.running = True

        # Data parallel routing state
        self.dp_request_counter = 0  # Round-robin counter for request routing

        # Start scheduler process(es)
        self._start_schedulers()

        # Wait a moment for schedulers to initialize
        time.sleep(2)

        # Initialize ZMQ context and sockets
        # Always use rank suffix format for consistency
        self.context = zmq.Context()
        self.request_sockets = []
        self.result_socket = self.context.socket(zmq.PULL)

        # Connect to all scheduler sockets (even if just one)
        for rank in range(self.dp_size):
            req_socket = self.context.socket(zmq.PUSH)
            req_socket.connect(f"ipc://{self.request_socket_path}_{rank}")
            self.request_sockets.append(req_socket)

        # Bind result socket (schedulers connect to us)
        self.result_socket.bind(f"ipc://{result_socket_path}")

        # Set socket options: lower HWMs to surface backpressure earlier
        try:
            for req_socket in self.request_sockets:
                req_socket.setsockopt(zmq.SNDHWM, 256)
                req_socket.setsockopt(zmq.LINGER, 0)
            self.result_socket.setsockopt(zmq.RCVHWM, 1024)
            self.result_socket.setsockopt(zmq.LINGER, 0)
        except Exception:
            pass

        # Create bounded in-process queue and sender thread to avoid handler blocking on ZMQ
        # Scale queue size with dp_size to avoid bottleneck with multiple replicas
        self.to_scheduler: queue.Queue[bytes] = queue.Queue(maxsize=max(1, self.max_batch_size * 2 * self.dp_size))
        self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.sender_thread.start()

        # Start background message processing thread
        self.message_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.message_thread.start()

        # Register cleanup on exit
        atexit.register(self.cleanup)

    def _start_schedulers(self):
        """Start the scheduler process(es)."""
        try:
            import subprocess
            import sys

            if self.dp_size > 1:
                # Data parallel mode: use subprocess to set CUDA_VISIBLE_DEVICES before Python starts
                self.scheduler_processes = []

                # Parse existing CUDA_VISIBLE_DEVICES mask if present
                import os

                existing_cuda_mask = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                if existing_cuda_mask is not None:
                    # User has pre-set a GPU mask, respect it
                    available_gpus = [int(x.strip()) for x in existing_cuda_mask.split(",") if x.strip().isdigit()]
                    if len(available_gpus) < self.dp_size:
                        self.logger.error(
                            f"CUDA_VISIBLE_DEVICES={existing_cuda_mask} provides {len(available_gpus)} GPUs, "
                            f"but --dp-size={self.dp_size} requires {self.dp_size} GPUs"
                        )
                        raise ValueError(f"Insufficient GPUs in CUDA_VISIBLE_DEVICES mask for dp_size={self.dp_size}")
                    self.logger.info(f"Using existing CUDA_VISIBLE_DEVICES mask: {existing_cuda_mask}")
                    gpu_mapping = available_gpus[: self.dp_size]
                else:
                    # No mask set, use 0, 1, 2, ... dp_size-1
                    gpu_mapping = list(range(self.dp_size))

                for rank in range(self.dp_size):
                    request_socket_path = f"{self.request_socket_path}_{rank}"
                    # All schedulers connect to the same result socket (no rank suffix)
                    result_socket_path = self.result_socket_path

                    # Create environment with CUDA_VISIBLE_DEVICES set to the mapped GPU
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_mapping[rank])

                    # Build command to run the scheduler entry point
                    cmd = [
                        sys.executable,
                        "-m",
                        "vox_serve.scheduler_entry",
                        "--dp-rank",
                        str(rank),
                        "--dp-size",
                        str(self.dp_size),
                        "--model-name",
                        self.model_name,
                        "--scheduler-type",
                        self.scheduler_type,
                        "--max-batch-size",
                        str(self.max_batch_size),
                        "--page-size",
                        str(self.page_size),
                        "--request-socket-path",
                        request_socket_path,
                        "--result-socket-path",
                        result_socket_path,
                        "--log-level",
                        get_global_log_level(),
                    ]

                    # Add optional parameters
                    if self.max_num_pages is not None:
                        cmd.extend(["--max-num-pages", str(self.max_num_pages)])
                    if self.top_p is not None:
                        cmd.extend(["--top-p", str(self.top_p)])
                    if self.top_k is not None:
                        cmd.extend(["--top-k", str(self.top_k)])
                    if self.min_p is not None:
                        cmd.extend(["--min-p", str(self.min_p)])
                    if self.temperature is not None:
                        cmd.extend(["--temperature", str(self.temperature)])
                    if self.max_tokens is not None:
                        cmd.extend(["--max-tokens", str(self.max_tokens)])
                    if self.repetition_penalty is not None:
                        cmd.extend(["--repetition-penalty", str(self.repetition_penalty)])
                    if self.repetition_window is not None:
                        cmd.extend(["--repetition-window", str(self.repetition_window)])
                    if self.cfg_scale is not None:
                        cmd.extend(["--cfg-scale", str(self.cfg_scale)])
                    if self.greedy:
                        cmd.append("--greedy")
                    if self.enable_cuda_graph:
                        cmd.append("--enable-cuda-graph")
                    if self.enable_disaggregation:
                        cmd.append("--enable-disaggregation")
                    if self.enable_nvtx:
                        cmd.append("--enable-nvtx")
                    if self.enable_torch_compile:
                        cmd.append("--enable-torch-compile")
                    if self.unroll_depth_cuda_graph:
                        cmd.append("--unroll-depth-cuda-graph")
                    if self.async_scheduling:
                        cmd.append("--async-scheduling")
                    if self.detokenize_interval is not None:
                        cmd.extend(["--detokenize-interval", str(self.detokenize_interval)])

                    self.logger.info(f"Starting DP rank {rank} with CUDA_VISIBLE_DEVICES={gpu_mapping[rank]}")
                    process = subprocess.Popen(cmd, env=env)
                    self.scheduler_processes.append(process)
                    self.logger.info(
                        f"Started scheduler process (DP rank {rank}/{self.dp_size}) with PID: {process.pid}"
                    )
            else:
                # Single scheduler mode - use subprocess for consistency
                self.scheduler_processes = None

                # Use rank 0 with suffix for request, but no suffix for result
                request_socket_path = f"{self.request_socket_path}_0"
                result_socket_path = self.result_socket_path

                # Build command to run the scheduler entry point
                cmd = [
                    sys.executable,
                    "-m",
                    "vox_serve.scheduler_entry",
                    "--dp-rank",
                    "0",
                    "--dp-size",
                    "1",
                    "--model-name",
                    self.model_name,
                    "--scheduler-type",
                    self.scheduler_type,
                    "--max-batch-size",
                    str(self.max_batch_size),
                    "--page-size",
                    str(self.page_size),
                    "--request-socket-path",
                    request_socket_path,
                    "--result-socket-path",
                    result_socket_path,
                    "--log-level",
                    get_global_log_level(),
                ]

                # Add optional parameters
                if self.max_num_pages is not None:
                    cmd.extend(["--max-num-pages", str(self.max_num_pages)])
                if self.top_p is not None:
                    cmd.extend(["--top-p", str(self.top_p)])
                if self.top_k is not None:
                    cmd.extend(["--top-k", str(self.top_k)])
                if self.min_p is not None:
                    cmd.extend(["--min-p", str(self.min_p)])
                if self.temperature is not None:
                    cmd.extend(["--temperature", str(self.temperature)])
                if self.max_tokens is not None:
                    cmd.extend(["--max-tokens", str(self.max_tokens)])
                if self.repetition_penalty is not None:
                    cmd.extend(["--repetition-penalty", str(self.repetition_penalty)])
                if self.repetition_window is not None:
                    cmd.extend(["--repetition-window", str(self.repetition_window)])
                if self.cfg_scale is not None:
                    cmd.extend(["--cfg-scale", str(self.cfg_scale)])
                if self.greedy:
                    cmd.append("--greedy")
                if self.enable_cuda_graph:
                    cmd.append("--enable-cuda-graph")
                if self.enable_disaggregation:
                    cmd.append("--enable-disaggregation")
                if self.enable_nvtx:
                    cmd.append("--enable-nvtx")
                if self.enable_torch_compile:
                    cmd.append("--enable-torch-compile")
                if self.unroll_depth_cuda_graph:
                    cmd.append("--unroll-depth-cuda-graph")
                if self.async_scheduling:
                    cmd.append("--async-scheduling")
                if self.detokenize_interval is not None:
                    cmd.extend(["--detokenize-interval", str(self.detokenize_interval)])

                process = subprocess.Popen(cmd)
                self.scheduler_process = process
                self.logger.info(f"Started scheduler process with PID: {process.pid}")

        except Exception as e:
            self.logger.error(f"Failed to start scheduler: {e}")
            raise RuntimeError(f"Could not start scheduler process: {e}") from e

    def _process_messages(self):
        """Process incoming scheduler messages in a background thread."""
        while self.running:
            try:
                while True:
                    # Use NOBLOCK to prevent message loss and add small sleep for efficiency
                    message = self.result_socket.recv(flags=zmq.NOBLOCK)

                    # Parse message format: request_id|TYPE|data
                    parts = message.split(b"|", 2)
                    if len(parts) >= 3:
                        request_id = parts[0].decode("utf-8")
                        message_type = parts[1].decode("utf-8")
                        data = parts[2]
                    else:
                        self.logger.warning(f"Malformed message received: {message[:100]}...")
                        continue

                    # Route message to the appropriate request
                    with self.request_lock:
                        # prune expired entries in recently_completed
                        if self.recently_completed:
                            now = time.time()
                            # pop from left while expired
                            to_pop = []
                            for rid, ts in self.recently_completed.items():
                                if now - ts > self.recently_completed_ttl_sec:
                                    to_pop.append(rid)
                                else:
                                    break
                            for rid in to_pop:
                                self.recently_completed.pop(rid, None)

                        if request_id in self.pending_requests:
                            if message_type == "AUDIO":
                                # Handle audio chunk
                                self.pending_requests[request_id]["chunks"].append(data)
                            elif message_type == "COMPLETION":
                                # Handle completion notification
                                completion_info = json.loads(data.decode("utf-8"))
                                self.logger.info(f"Request {request_id} completed: {completion_info}")
                                self.pending_requests[request_id]["event"].set()
                                # remember completion to suppress late messages
                                self.recently_completed[request_id] = time.time()
                        # If we've very recently completed this request, drop silently (debug only)
                        elif request_id in self.recently_completed:
                            self.logger.debug(
                                f"Dropping late {message_type} for recently completed request {request_id}"
                            )
                        else:
                            # Log when we receive messages for unknown requests
                            self.logger.warning(f"Received {message_type} message for unknown request {request_id}")

            except zmq.Again:
                # No message available, sleep briefly to avoid busy waiting
                time.sleep(0.001)
                continue
            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    self.logger.error(f"Error in message processing: {e}")
                continue

    def _stop_scheduler(self):
        """Stop scheduler process(es) if they are running."""
        if self.dp_size > 1:
            # Stop all scheduler processes in DP mode
            if self.scheduler_processes:
                self.logger.info(f"Stopping {self.dp_size} scheduler processes...")
                for i, process in enumerate(self.scheduler_processes):
                    if process.poll() is None:  # Process is still running
                        try:
                            process.terminate()
                            try:
                                process.wait(timeout=1)
                            except subprocess.TimeoutExpired:
                                self.logger.warning(f"Scheduler {i} didn't terminate gracefully, forcing kill...")
                                process.kill()
                                process.wait(timeout=1)
                        except Exception as e:
                            self.logger.error(f"Error stopping scheduler {i}: {e}")
                self.logger.info("All scheduler processes stopped")
        elif hasattr(self, "scheduler_process") and self.scheduler_process and self.scheduler_process.poll() is None:
            self.logger.info("Stopping scheduler process...")
            try:
                self.scheduler_process.terminate()
                try:
                    self.scheduler_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Scheduler didn't terminate gracefully, forcing kill...")
                    self.scheduler_process.kill()
                    self.scheduler_process.wait(timeout=1)
            except Exception as e:
                self.logger.error(f"Error stopping scheduler: {e}")
            self.logger.info("Scheduler process stopped")

    def _enqueue_request(self, payload: bytes) -> None:
        """Enqueue a request payload to be forwarded to the scheduler.

        Refuses when saturated to keep latency bounded.
        """
        try:
            self.to_scheduler.put_nowait(payload)
        except queue.Full:
            raise HTTPException(status_code=429, detail="Server busy; please retry shortly") from None

    def _sender_loop(self) -> None:
        """Dedicated thread that drains the in-process queue and sends to ZMQ without blocking the handler."""
        backoff_initial = 0.001
        backoff_max = 0.02
        while self.running:
            try:
                try:
                    payload = self.to_scheduler.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Select target socket for data parallel routing (round-robin)
                # Pin this request to the selected rank even under backpressure
                target_socket = self.request_sockets[self.dp_request_counter % self.dp_size]
                self.dp_request_counter += 1

                backoff = backoff_initial
                while self.running:
                    try:
                        # Non-blocking send; back off briefly on ZMQ backpressure
                        target_socket.send(payload, flags=zmq.DONTWAIT)
                        break
                    except zmq.Again:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, backoff_max)
                    except Exception as e:
                        self.logger.error(f"Sender loop error during send: {e}")
                        break
            except Exception as e:
                if self.running:
                    self.logger.error(f"Sender loop error: {e}")

    def start_streaming_request(
        self,
        text: str = None,
        audio_path: str = None,
        model_kwargs: Dict = None,
    ) -> str:
        """Create and enqueue a streaming request, returning its request ID.

        Args:
            text: Input text to synthesize.
            audio_path: Optional path to input audio for STS-capable models.
            model_kwargs: Optional model-specific parameters (e.g., language, speaker).

        Returns:
            The request ID used for subsequent streaming.
        """
        request_id = str(uuid.uuid4())
        self.logger.info(f"Request {request_id} joined for streaming")

        # Register this request for concurrent processing
        completion_event = threading.Event()
        with self.request_lock:
            self.pending_requests[request_id] = {
                "chunks": [],
                "event": completion_event,
                "streaming": True,
                "consumed_chunks": 0,
            }

        # Serialize and send to scheduler
        request_dict = {
            "request_id": request_id,
            "prompt": text,
            "audio_path": audio_path,
            "is_streaming": True,
            "model_kwargs": model_kwargs or {},
        }
        request_json = json.dumps(request_dict)
        message = f"{request_json}|audio_data_placeholder".encode("utf-8")
        self._enqueue_request(message)

        return request_id

    def start_input_streaming_request(
        self,
        audio_path: str = None,
        model_kwargs: Dict = None,
    ) -> str:
        """Create and enqueue an input streaming request (text will be sent incrementally).

        Args:
            audio_path: Optional path to input audio for voice cloning.
            model_kwargs: Optional model-specific parameters (e.g., language, speaker).

        Returns:
            The request ID used for subsequent text chunks.
        """
        request_id = str(uuid.uuid4())
        self.logger.info(f"Request {request_id} joined for input streaming")

        # Register this request for concurrent processing
        completion_event = threading.Event()
        with self.request_lock:
            self.pending_requests[request_id] = {
                "chunks": [],
                "event": completion_event,
                "streaming": True,
                "consumed_chunks": 0,
                "input_streaming": True,  # Mark as input streaming
            }

        # Send TEXT_STREAM_START message to scheduler
        # Format: request_id|TEXT_STREAM_START|{json_config}
        config = {
            "audio_path": audio_path,
            "is_streaming": True,
            "model_kwargs": model_kwargs or {},
        }
        message = f"{request_id}|TEXT_STREAM_START|{json.dumps(config)}".encode("utf-8")
        self._enqueue_request(message)

        return request_id

    def send_text_chunk(self, request_id: str, text: str) -> bool:
        """Send a text chunk for an input streaming request.

        Args:
            request_id: Request identifier returned by ``start_input_streaming_request``.
            text: Text chunk to add to the generation.

        Returns:
            True if the text was sent successfully.

        Raises:
            HTTPException: If the request is not found or already completed.
        """
        with self.request_lock:
            request_data = self.pending_requests.get(request_id)
            if not request_data:
                raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
            if request_data["event"].is_set():
                raise HTTPException(status_code=400, detail=f"Request {request_id} already completed")

        # Send TEXT_UPDATE message to scheduler
        # Format: request_id|TEXT_UPDATE|text_chunk
        message = f"{request_id}|TEXT_UPDATE|{text}".encode("utf-8")
        self._enqueue_request(message)
        self.logger.debug(f"Sent text chunk for request {request_id}: {len(text)} chars")
        return True

    def end_input_streaming(self, request_id: str) -> None:
        """Signal end of text input for an input streaming request.

        Args:
            request_id: Request identifier returned by ``start_input_streaming_request``.

        Raises:
            HTTPException: If the request is not found.
        """
        with self.request_lock:
            request_data = self.pending_requests.get(request_id)
            if not request_data:
                raise HTTPException(status_code=404, detail=f"Request {request_id} not found")

        # Send TEXT_COMPLETE message to scheduler
        # Format: request_id|TEXT_COMPLETE|
        message = f"{request_id}|TEXT_COMPLETE|".encode("utf-8")
        self._enqueue_request(message)
        self.logger.info(f"Text input complete for request {request_id}")

    async def async_stream_chunks(self, request_id: str):
        """Yield audio chunks for an already enqueued streaming request.

        Args:
            request_id: Request identifier returned by ``start_streaming_request``.

        Yields:
            Raw audio chunks (bytes) as they arrive.

        Raises:
            HTTPException: If the request times out or fails.
        """
        start_time = time.time()
        # Stream chunks until completion
        while True:
            # Timeout check
            if time.time() - start_time > self.timeout_seconds:
                # Cleanup on timeout
                with self.request_lock:
                    self.pending_requests.pop(request_id, None)
                raise HTTPException(status_code=500, detail="Generation timed out")

            new_chunks: list[bytes] = []
            done = False
            with self.request_lock:
                request_data = self.pending_requests.get(request_id)
                if request_data:
                    available = len(request_data["chunks"])
                    consumed = request_data.get("consumed_chunks", 0)
                    new_chunks = request_data["chunks"][consumed:available]
                    request_data["consumed_chunks"] = available
                    done = request_data["event"].is_set()
                else:
                    # No request found; treat as done
                    done = True

            for chunk in new_chunks:
                yield chunk

            if done:
                # Yield any remaining chunks and cleanup
                remaining: list[bytes] = []
                with self.request_lock:
                    request_data = self.pending_requests.get(request_id)
                    if request_data:
                        consumed = request_data.get("consumed_chunks", 0)
                        remaining = request_data["chunks"][consumed:]
                        self.pending_requests.pop(request_id, None)
                for chunk in remaining:
                    yield chunk
                break

            # Small async sleep to avoid busy-waiting
            await asyncio.sleep(0.001)

    def generate_audio(
        self,
        text: str = None,
        audio_path: str = None,
        model_kwargs: Dict = None,
        sample_rate: int = None,
    ) -> str:
        """
        Generate audio from text and return path to the audio file.

        Args:
            text: Input text to synthesize (optional if audio_path provided)
            audio_path: Path to input audio file (optional)
            model_kwargs: Optional model-specific parameters (e.g., language, speaker).
            sample_rate: Output sample rate (8000, 16000, or 24000). Defaults to native 24 kHz.

        Returns:
            Path to the generated audio file

        Raises:
            HTTPException: If generation fails or times out
        """
        request_id = str(uuid.uuid4())
        self.logger.info(f"Request {request_id} joined for generation")
        out_sr = sample_rate if sample_rate is not None else _NATIVE_SAMPLE_RATE

        # Register this request for concurrent processing
        completion_event = threading.Event()
        with self.request_lock:
            self.pending_requests[request_id] = {"chunks": [], "event": completion_event}

        try:
            # Serialize Request object to JSON
            request_dict = {
                "request_id": request_id,
                "prompt": text,
                "audio_path": audio_path,
                "is_streaming": False,
                "model_kwargs": model_kwargs or {},
            }

            request_json = json.dumps(request_dict)
            message = f"{request_json}|audio_data_placeholder".encode("utf-8")

            # Enqueue request to scheduler (non-blocking)
            self._enqueue_request(message)

            # Wait for completion or timeout
            if not completion_event.wait(timeout=self.timeout_seconds):
                raise HTTPException(status_code=500, detail="Generation timed out")

            # Retrieve collected audio chunks
            with self.request_lock:
                audio_chunks = self.pending_requests[request_id]["chunks"][:]
                del self.pending_requests[request_id]

            if not audio_chunks:
                raise HTTPException(status_code=500, detail="No audio generated")

            # Optionally resample native 24 kHz PCM to the requested rate
            if out_sr != _NATIVE_SAMPLE_RATE:
                resampler = PcmResampler(_NATIVE_SAMPLE_RATE, out_sr)
                audio_chunks = [c for c in (resampler.process(c) for c in audio_chunks) if c]

            # Save to WAV file
            output_file = self.output_dir / f"{request_id}.wav"
            with wave.open(str(output_file), "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(out_sr)
                for chunk in audio_chunks:
                    wf.writeframes(chunk)

            return str(output_file)

        except Exception as e:
            # Clean up on error
            with self.request_lock:
                self.pending_requests.pop(request_id, None)
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}") from e

    def cleanup(self):
        """Clean up ZMQ resources, background threads, and scheduler processes."""
        self.logger.info("Cleaning up API server...")

        # Stop background message processing
        self.running = False
        if hasattr(self, "message_thread") and self.message_thread.is_alive():
            self.message_thread.join(timeout=1)
        if hasattr(self, "sender_thread") and self.sender_thread.is_alive():
            self.sender_thread.join(timeout=1)

        try:
            # Close all request sockets
            if hasattr(self, "request_sockets"):
                for req_socket in self.request_sockets:
                    req_socket.close()
            if hasattr(self, "result_socket"):
                self.result_socket.close()
            if hasattr(self, "context"):
                self.context.term()  # Terminate ZMQ context
        except Exception as e:
            self.logger.error(f"Error cleaning up ZMQ: {e}")

        self._stop_scheduler()


# Initialize FastAPI app
app = FastAPI(title="Vox-Serve API", description="Text-to-Speech API using Orpheus model")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Global API server instance (will be initialized in main)
api_server = None


@app.post("/generate")
async def generate(
    text: str = Form(...),
    audio: Optional[UploadFile] = File(None),
    voice_id: Optional[str] = Form(None),
    streaming: bool = Form(True),
    sample_rate: Optional[int] = Form(None),
    # Model-specific parameters (used by models like Qwen3-TTS)
    language: Optional[str] = Form(None),
    speaker: Optional[str] = Form(None),
    ref_text: Optional[str] = Form(None),
    instruct: Optional[str] = Form(None),
    x_vector_only_mode: Optional[bool] = Form(None),
):
    """
    Generate speech from text and return audio file or streaming response.

    Args:
        text: Input text to synthesize
        audio: Optional input audio file
        streaming: Whether to return streaming response (default: True)
        sample_rate: Output sample rate in Hz (8000, 16000, or 24000; default 24000)
        language: Language code for synthesis (model-specific, e.g., "en", "zh", "auto")
        speaker: Speaker ID for multi-speaker models (model-specific)
        ref_text: Reference text for voice cloning (used with audio for ICL mode)
        instruct: Instruction text for voice design/control (model-specific)
        x_vector_only_mode: If True, use only speaker embedding without ICL (model-specific)

    Returns:
        Audio file as direct response (if streaming=False) or streaming audio response (if streaming=True)
    """
    raise HTTPException(
        status_code=410,
        detail="/generate is disabled. Use WebSocket /ws for synthesis.",
    )

    if api_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    try:
        out_sr = _parse_sample_rate(sample_rate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    audio_path = None
    is_temp_audio = False  # only temp uploads get cleaned up; registered voices persist
    if audio:
        # Save uploaded audio file
        audio_filename = f"{uuid.uuid4()}_{audio.filename}"
        audio_path = str(api_server.upload_dir / audio_filename)

        # Move file write off the event loop to avoid blocking under load
        content = await audio.read()
        await run_in_threadpool(Path(audio_path).write_bytes, content)
        is_temp_audio = True
    elif voice_id:
        # Resolve a previously registered voice (register-once, no upload)
        with api_server.voices_lock:
            audio_path = api_server.voices.get(voice_id)
        if audio_path is None or not Path(audio_path).exists():
            raise HTTPException(status_code=404, detail=f"Unknown voice_id: {voice_id}")

    # Build model-specific kwargs (only include non-None values)
    model_kwargs = {}
    if language is not None:
        model_kwargs["language"] = language
    if speaker is not None:
        model_kwargs["speaker"] = speaker
    if ref_text is not None:
        model_kwargs["ref_text"] = ref_text
    if instruct is not None:
        model_kwargs["instruct"] = instruct
    if x_vector_only_mode is not None:
        model_kwargs["x_vector_only_mode"] = x_vector_only_mode

    try:
        if streaming:
            # Streaming response: enqueue request immediately, then stream asynchronously
            request_id = api_server.start_streaming_request(text, audio_path, model_kwargs)

            async def audio_stream():
                # WAV header for mono 16-bit audio at the requested sample rate
                wav_header = io.BytesIO()
                with wave.open(wav_header, "wb") as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(out_sr)
                    wf.writeframes(b"")  # Empty data for header

                # Get header bytes and correct the chunk size for streaming
                wav_header.seek(0)
                header_bytes = wav_header.read()

                # Send WAV header first
                yield header_bytes

                resampler = (
                    PcmResampler(_NATIVE_SAMPLE_RATE, out_sr)
                    if out_sr != _NATIVE_SAMPLE_RATE
                    else None
                )

                # Stream audio chunks asynchronously
                async for chunk in api_server.async_stream_chunks(request_id):
                    if resampler is not None:
                        chunk = resampler.process(chunk)
                    if chunk:
                        yield chunk

            return StreamingResponse(
                audio_stream(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=stream_{uuid.uuid4().hex[:8]}.wav",
                    "Cache-Control": "no-cache",
                },
            )
        else:
            # Non-streaming response
            audio_file = await run_in_threadpool(
                api_server.generate_audio, text, audio_path, model_kwargs, out_sr
            )
            request_id = Path(audio_file).stem

            return FileResponse(path=audio_file, media_type="audio/wav", filename=f"{request_id}.wav")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        # Schedule cleanup of uploaded file after a delay to ensure processing is complete.
        # Registered voices (voice_id) are persistent and must never be deleted here.
        if is_temp_audio and audio_path and Path(audio_path).exists():

            def delayed_cleanup():
                import time

                time.sleep(60)  # Wait 60 seconds before cleanup
                if Path(audio_path).exists():
                    Path(audio_path).unlink()

            cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
            cleanup_thread.start()


# ============================================================================
# Voice registry endpoints (register-once reference audio)
# ============================================================================


@app.post("/voices")
async def register_voice(
    audio: UploadFile = File(...),
    voice_id: Optional[str] = Form(None),
):
    """Register a reference voice once and get back a ``voice_id``.

    Upload the reference audio a single time here, then pass ``voice_id`` on
    subsequent ``/generate`` calls instead of re-uploading the file each request.

    Args:
        audio: Reference audio file (WAV recommended).
        voice_id: Optional explicit id; if omitted a UUID is generated. Reusing
            an existing id overwrites that voice.

    Returns:
        ``{"voice_id": ..., "bytes": ...}``
    """
    if api_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    vid = voice_id or uuid.uuid4().hex[:12]
    # Preserve the original extension so downstream loaders behave the same
    ext = Path(audio.filename or "ref.wav").suffix or ".wav"
    stored_path = str(api_server.voices_dir / f"{vid}{ext}")

    content = await audio.read()
    await run_in_threadpool(Path(stored_path).write_bytes, content)

    with api_server.voices_lock:
        # If overwriting an id that had a different extension, drop the old file
        old = api_server.voices.get(vid)
        if old and old != stored_path and Path(old).exists():
            Path(old).unlink()
        api_server.voices[vid] = stored_path

    return {"voice_id": vid, "bytes": len(content)}


@app.get("/voices")
async def list_voices():
    """List registered voice ids."""
    if api_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    with api_server.voices_lock:
        return {"voices": sorted(api_server.voices.keys())}


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a registered voice."""
    if api_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    with api_server.voices_lock:
        path = api_server.voices.pop(voice_id, None)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown voice_id: {voice_id}")
    if Path(path).exists():
        Path(path).unlink()
    return {"deleted": voice_id}


# ============================================================================
# WebSocket streaming endpoint (persistent connection, many utterances)
# ============================================================================

# Model workers emit int16 PCM mono at this native rate. Clients may request a
# lower rate (8 kHz telephony, 16 kHz) via sample_rate; the API resamples on the
# way out. All supported rates are valid Opus input rates too.
_NATIVE_SAMPLE_RATE = 24000
_SUPPORTED_SAMPLE_RATES = frozenset({8000, 16000, 24000})
_OPUS_FRAME_MS = 20
# 48 kbps mono is transparent for speech with headroom; still ~8x smaller than
# raw PCM (384 kbps @ 24 kHz). Override via VOX_OPUS_BITRATE.
_OPUS_BITRATE = int(os.environ.get("VOX_OPUS_BITRATE", "48000"))


def _parse_sample_rate(value) -> int:
    """Validate and normalize a sample_rate request (default: native 24 kHz)."""
    if value is None or value == "":
        return _NATIVE_SAMPLE_RATE
    try:
        sr = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid sample_rate {value!r}; supported: "
            f"{sorted(_SUPPORTED_SAMPLE_RATES)}"
        ) from e
    if sr not in _SUPPORTED_SAMPLE_RATES:
        raise ValueError(
            f"Unsupported sample_rate={sr}; supported: "
            f"{sorted(_SUPPORTED_SAMPLE_RATES)}"
        )
    return sr


class PcmResampler:
    """Streaming int16 mono resampler (stateful across chunks)."""

    def __init__(self, src_rate: int, dst_rate: int):
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._state = None

    def process(self, pcm_bytes: bytes) -> bytes:
        if not pcm_bytes or self.src_rate == self.dst_rate:
            return pcm_bytes
        converted, self._state = audioop.ratecv(
            pcm_bytes, 2, 1, self.src_rate, self.dst_rate, self._state
        )
        return converted


class OpusStreamEncoder:
    """Incremental Opus encoder for a stream of int16 PCM mono.

    Buffers PCM into fixed 20 ms frames and emits one raw Opus packet per
    frame. Raw packets (no container) keep latency minimal and let clients
    decode with WebCodecs ``AudioDecoder`` or any libopus binding. Each packet
    is sent as its own WebSocket binary frame, so message boundaries already
    delimit packets (no length prefix needed).
    """

    def __init__(self, sample_rate: int = _NATIVE_SAMPLE_RATE):
        import opuslib

        self.sample_rate = sample_rate
        self._frame_samples = sample_rate * _OPUS_FRAME_MS // 1000
        self._frame_bytes = self._frame_samples * 2  # 16-bit mono
        self._enc = opuslib.Encoder(sample_rate, 1, opuslib.APPLICATION_AUDIO)
        try:
            self._enc.bitrate = _OPUS_BITRATE
        except Exception:  # noqa: BLE001 - fall back to libopus auto bitrate
            pass
        self._buf = bytearray()

    def encode(self, pcm_bytes: bytes) -> list[bytes]:
        """Feed PCM bytes; return zero or more complete Opus packets."""
        self._buf.extend(pcm_bytes)
        packets = []
        while len(self._buf) >= self._frame_bytes:
            frame = bytes(self._buf[: self._frame_bytes])
            del self._buf[: self._frame_bytes]
            packets.append(self._enc.encode(frame, self._frame_samples))
        return packets

    def flush(self) -> list[bytes]:
        """Encode any leftover tail, zero-padded to a full frame."""
        if not self._buf:
            return []
        frame = bytes(self._buf) + b"\x00" * (self._frame_bytes - len(self._buf))
        self._buf.clear()
        return [self._enc.encode(frame, self._frame_samples)]


def _resolve_audio_and_kwargs(msg: dict):
    """Resolve reference audio path and model kwargs from a WS request dict.

    Returns ``(audio_path, is_temp_audio, model_kwargs)``. Supports a
    pre-registered ``voice_id`` (no upload, preferred) or inline
    base64-encoded ``audio``/``audio_base64`` for one-off references.
    """
    audio_path = None
    is_temp_audio = False

    voice_id = msg.get("voice_id")
    audio_b64 = msg.get("audio_base64") or msg.get("audio")
    if voice_id:
        with api_server.voices_lock:
            audio_path = api_server.voices.get(voice_id)
        if audio_path is None or not Path(audio_path).exists():
            raise ValueError(f"Unknown voice_id: {voice_id}")
    elif audio_b64:
        raw = base64.b64decode(audio_b64)
        audio_path = str(api_server.upload_dir / f"{uuid.uuid4().hex}.wav")
        Path(audio_path).write_bytes(raw)
        is_temp_audio = True

    model_kwargs = {}
    for key in (
        "language", "speaker", "ref_text", "instruct", "x_vector_only_mode",
        # Zonos conditioning controls
        "speaking_rate", "pitch_std", "emotion",
    ):
        if msg.get(key) is not None:
            model_kwargs[key] = msg[key]
    if voice_id:
        model_kwargs["voice_id"] = voice_id

    return audio_path, is_temp_audio, model_kwargs


@app.websocket("/ws")
async def ws_generate(websocket: WebSocket):
    """Persistent WebSocket for low-latency streaming synthesis.

    Open the socket once and reuse it for many utterances, avoiding the
    per-request TCP+TLS handshake that dominates latency over long distances.

    Client -> server (JSON text frame per utterance)::

        {"text": "Hello world", "voice_id": "umair", "language": "en"}

    Optional one-off reference instead of voice_id: ``"audio_base64": "<wav>"``.
    Set ``"format": "opus"`` for ~10x smaller frames (transparent for speech);
    defaults to raw ``"pcm"``. Optional ``"sample_rate"`` of 8000, 16000, or
    24000 (default) resamples output; Opus encodes at the requested rate. Send
    ``{"type": "close"}`` to end the session.

    Zonos conditioning controls (all optional): ``"speaking_rate"`` (phonemes/min,
    ~15 normal), ``"pitch_std"`` (expressiveness, 20-45 normal / 60-150 expressive),
    and ``"emotion"`` (8-value list [happiness, sadness, disgust, fear, surprise,
    anger, other, neutral]).

    Server -> client, per utterance::

        {"type": "start", "request_id": ..., "sample_rate": 8000|16000|24000,
         "channels": 1, "format": "pcm_s16le" | "opus", "frame_ms": 20}
        <binary frame> <binary frame> ...   # PCM chunks, or one Opus packet each
        {"type": "end", "request_id": ...}          # JSON text frame

    Errors are reported as ``{"type": "error", "detail": ...}`` without
    closing the socket, so the client can continue with the next utterance.
    """
    await websocket.accept()
    if api_server is None:
        await websocket.send_json({"type": "error", "detail": "Server not ready"})
        await websocket.close()
        return

    try:
        while True:
            msg = await websocket.receive_json()

            if msg.get("type") == "close":
                break

            text = msg.get("text")
            if not text:
                await websocket.send_json({"type": "error", "detail": "Missing 'text'"})
                continue

            audio_path = None
            is_temp_audio = False
            request_id = None
            use_opus = str(msg.get("format", "pcm")).lower() == "opus"
            try:
                out_sr = _parse_sample_rate(msg.get("sample_rate"))
                audio_path, is_temp_audio, model_kwargs = _resolve_audio_and_kwargs(msg)
                request_id = api_server.start_streaming_request(text, audio_path, model_kwargs)

                resampler = (
                    PcmResampler(_NATIVE_SAMPLE_RATE, out_sr)
                    if out_sr != _NATIVE_SAMPLE_RATE
                    else None
                )
                opus_enc = OpusStreamEncoder(out_sr) if use_opus else None
                await websocket.send_json({
                    "type": "start",
                    "request_id": request_id,
                    "sample_rate": out_sr,
                    "channels": 1,
                    "format": "opus" if use_opus else "pcm_s16le",
                    "frame_ms": _OPUS_FRAME_MS if use_opus else None,
                })

                async for chunk in api_server.async_stream_chunks(request_id):
                    if not chunk:
                        continue
                    if resampler is not None:
                        chunk = resampler.process(chunk)
                        if not chunk:
                            continue
                    if opus_enc is not None:
                        for pkt in opus_enc.encode(chunk):
                            await websocket.send_bytes(pkt)
                    else:
                        await websocket.send_bytes(chunk)

                if opus_enc is not None:
                    for pkt in opus_enc.flush():
                        await websocket.send_bytes(pkt)

                await websocket.send_json({"type": "end", "request_id": request_id})
            except ValueError as e:
                await websocket.send_json({"type": "error", "detail": str(e)})
            except Exception as e:  # noqa: BLE001
                api_server.logger.error(f"WS synth error: {e}")
                await websocket.send_json({"type": "error", "detail": str(e), "request_id": request_id})
            finally:
                if is_temp_audio and audio_path and Path(audio_path).exists():
                    try:
                        Path(audio_path).unlink()
                    except OSError:
                        pass
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        api_server.logger.error(f"WS connection error: {e}")
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


# ============================================================================
# OpenAI-compatible /v1/audio/speech endpoint (raw PCM streaming)
# ============================================================================


class AudioSpeechRequest(BaseModel):
    """Request body for the OpenAI-compatible ``/v1/audio/speech`` endpoint.

    Mirrors the payload sent by ``benchmarking/bench/adapters/voxtral.py``:
    ``{model, input, voice, language, response_format, stream, extra_params}``.
    ``extra_params`` may carry ``cfg_alpha`` (classifier-free guidance scale)
    or ``sample_rate`` (8000, 16000, or 24000).
    """

    model: Optional[str] = None
    input: str
    voice: Optional[str] = None
    language: Optional[str] = None
    response_format: Optional[str] = "pcm"
    stream: Optional[bool] = True
    sample_rate: Optional[int] = None
    extra_params: Optional[Dict] = None


@app.post("/v1/audio/speech")
async def audio_speech(req: AudioSpeechRequest):
    """Generate speech and stream raw int16 PCM (no WAV header).

    Reuses the existing streaming machinery (``async_stream_chunks``); the worker
    already emits int16 PCM bytes, so unlike the ``/generate`` WAV route this
    endpoint forwards them (optionally resampled) with ``media_type="audio/pcm"``.
    """
    if api_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    try:
        out_sr = _parse_sample_rate(
            req.sample_rate
            if req.sample_rate is not None
            else (req.extra_params or {}).get("sample_rate")
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    model_kwargs = {
        "voice": req.voice,
        "language": req.language,
        "cfg_alpha": (req.extra_params or {}).get("cfg_alpha"),
    }

    try:
        request_id = api_server.start_streaming_request(req.input, None, model_kwargs)

        async def audio_stream():
            resampler = (
                PcmResampler(_NATIVE_SAMPLE_RATE, out_sr)
                if out_sr != _NATIVE_SAMPLE_RATE
                else None
            )
            async for chunk in api_server.async_stream_chunks(request_id):
                if resampler is not None:
                    chunk = resampler.process(chunk)
                if chunk:
                    yield chunk

        return StreamingResponse(
            audio_stream(),
            media_type="audio/pcm",
            headers={
                "Cache-Control": "no-cache",
                "X-Sample-Rate": str(out_sr),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Input Streaming Endpoints
# ============================================================================


@app.post("/generate/stream/start")
async def start_input_streaming(
    audio: Optional[UploadFile] = File(None),
    # Model-specific parameters
    language: Optional[str] = Form(None),
    speaker: Optional[str] = Form(None),
    ref_text: Optional[str] = Form(None),
    instruct: Optional[str] = Form(None),
    x_vector_only_mode: Optional[bool] = Form(None),
):
    """
    Start an input streaming request. Text will be sent incrementally via subsequent calls.

    Args:
        audio: Optional input audio file for voice cloning
        language: Language code for synthesis (model-specific)
        speaker: Speaker ID for multi-speaker models
        ref_text: Reference text for voice cloning
        instruct: Instruction text for voice design/control
        x_vector_only_mode: If True, use only speaker embedding without ICL

    Returns:
        JSON with request_id to use for subsequent text chunks
    """
    if api_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    audio_path = None
    if audio:
        # Save uploaded audio file
        audio_filename = f"{uuid.uuid4()}_{audio.filename}"
        audio_path = str(api_server.upload_dir / audio_filename)
        content = await audio.read()
        await run_in_threadpool(Path(audio_path).write_bytes, content)

    # Build model-specific kwargs
    model_kwargs = {}
    if language is not None:
        model_kwargs["language"] = language
    if speaker is not None:
        model_kwargs["speaker"] = speaker
    if ref_text is not None:
        model_kwargs["ref_text"] = ref_text
    if instruct is not None:
        model_kwargs["instruct"] = instruct
    if x_vector_only_mode is not None:
        model_kwargs["x_vector_only_mode"] = x_vector_only_mode

    try:
        request_id = api_server.start_input_streaming_request(audio_path, model_kwargs)
        return {"request_id": request_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/generate/stream/{request_id}/text")
async def send_text_chunk(
    request_id: str,
    text: str = Form(...),
):
    """
    Send a text chunk for an ongoing input streaming request.

    Args:
        request_id: Request identifier from start_input_streaming
        text: Text chunk to add to the generation

    Returns:
        JSON with status and request_id
    """
    if api_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    try:
        api_server.send_text_chunk(request_id, text)
        return {"status": "accepted", "request_id": request_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/generate/stream/{request_id}/audio")
async def stream_audio(request_id: str, sample_rate: Optional[int] = None):
    """
    Start streaming audio output for an input streaming request.

    Call this immediately after /start to receive audio chunks as they are generated,
    while continuing to send text via /text endpoint.

    Args:
        request_id: Request identifier from start_input_streaming
        sample_rate: Output sample rate in Hz (8000, 16000, or 24000; default 24000)

    Returns:
        Streaming audio response (WAV format)
    """
    if api_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    try:
        out_sr = _parse_sample_rate(sample_rate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Validate request exists
    with api_server.request_lock:
        request_data = api_server.pending_requests.get(request_id)
        if not request_data:
            raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
        if not request_data.get("input_streaming"):
            raise HTTPException(
                status_code=400,
                detail=f"Request {request_id} is not an input streaming request",
            )

    async def audio_stream():
        # WAV header for mono 16-bit audio at the requested sample rate
        wav_header = io.BytesIO()
        with wave.open(wav_header, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(out_sr)
            wf.writeframes(b"")  # Empty data for header

        wav_header.seek(0)
        header_bytes = wav_header.read()
        yield header_bytes

        resampler = (
            PcmResampler(_NATIVE_SAMPLE_RATE, out_sr)
            if out_sr != _NATIVE_SAMPLE_RATE
            else None
        )

        # Stream audio chunks asynchronously
        async for chunk in api_server.async_stream_chunks(request_id):
            if resampler is not None:
                chunk = resampler.process(chunk)
            if chunk:
                yield chunk

    return StreamingResponse(
        audio_stream(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename=stream_{request_id[:8]}.wav",
            "Cache-Control": "no-cache",
        },
    )


@app.post("/generate/stream/{request_id}/end")
async def end_input_streaming(request_id: str):
    """
    Signal end of text input for an input streaming request.

    This signals that no more text will be sent. If you're using the /audio endpoint
    to stream audio, that stream will complete after this is called.

    Args:
        request_id: Request identifier from start_input_streaming

    Returns:
        JSON confirmation
    """
    if api_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    try:
        # Signal text completion
        api_server.end_input_streaming(request_id)
        return {"status": "completed", "request_id": request_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


def _metrics_dir() -> Path:
    return Path(os.environ.get("VOX_METRICS_DIR", "/var/log/vox-serve"))


@app.get("/stats")
async def server_stats():
    """Live scheduler snapshot: active requests, latency percentiles, GPU + KV memory.

    Remote example: ``curl http://HOST:2200/stats``
    """
    stats_path = _metrics_dir() / "stats.json"
    if not stats_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"No metrics yet at {stats_path}. Wait a few seconds after the server starts.",
        )
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {e}") from e


@app.get("/stats/requests")
async def server_stats_requests(limit: int = 50):
    """Recent completed requests (from ``requests.jsonl``), newest last.

    Remote example: ``curl 'http://HOST:2200/stats/requests?limit=20'``
    """
    limit = max(1, min(int(limit), 1000))
    path = _metrics_dir() / "requests.jsonl"
    if not path.exists():
        return {"count": 0, "requests": []}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return {"count": len(rows), "requests": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read request log: {e}") from e


@app.get("/stats/log")
async def server_stats_log(lines: int = 40):
    """Tail of the human-readable metrics log.

    Remote example: ``curl 'http://HOST:2200/stats/log?lines=40'``
    """
    lines = max(1, min(int(lines), 500))
    path = _metrics_dir() / "stats.log"
    if not path.exists():
        return {"text": "", "path": str(path)}
    try:
        content = path.read_text(encoding="utf-8").splitlines()
        tail = "\n".join(content[-lines:])
        return {"text": tail, "path": str(path), "lines": min(lines, len(content))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read stats log: {e}") from e


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if api_server is not None:
        api_server.cleanup()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"\nReceived signal {signum}, shutting down...")
    if api_server is not None:
        api_server.cleanup()
    import os

    os._exit(0)  # Force immediate exit


def main():
    """Run the VoxServe API server from the CLI.

    The CLI maps to ``python -m vox_serve.launch`` or the ``vox-serve`` entrypoint.

    Arguments are parsed via ``argparse`` and control model selection, scheduling
    behavior, sampling parameters, and performance settings.
    """
    import argparse
    import multiprocessing as mp

    import uvicorn

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vox-Serve Text-to-Speech API Server")
    parser.add_argument(
        "--model",
        type=str,
        default="canopylabs/orpheus-3b-0.1-ft",
        help="Model name or path to use for text-to-speech synthesis (default: canopylabs/orpheus-3b-0.1-ft)",
    )
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="base",
        choices=["base", "online", "offline", "input_streaming"],
        help="Type of scheduler to use (default: base). Use 'input_streaming' for incremental text input.",
    )
    parser.add_argument("--async-scheduling", action="store_true", help="Enable async scheduling mode (default: False)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to (default: 8000)")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Maximum batch size for inference (default: 8)")
    parser.add_argument(
        "--max-num-pages", type=int, default=2048, help="Maximum number of KV cache pages (default: 1024)"
    )
    parser.add_argument("--page-size", type=int, default=128, help="Size of each KV cache page (default: 128)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling parameter (default: None)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling parameter (default: None)")
    parser.add_argument("--min-p", type=float, default=None, help="Min-p sampling parameter (default: None)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for sampling (default: None)")
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Maximum number of tokens to generate (default: model-specific)"
    )
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty (default: None)")
    parser.add_argument("--repetition-window", type=int, default=None, help="Repetition window size (default: None)")
    parser.add_argument("--cfg-scale", type=float, default=None, help="CFG scale for guidance (default: None)")
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Enable greedy sampling (ignores top-k, top-p, min-p, and temperature parameters)",
    )
    parser.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        default=True,
        help="Enable CUDA graph optimization for decode phase (default: True)",
    )
    parser.add_argument(
        "--disable-cuda-graph", action="store_true", help="Disable CUDA graph optimization for decode phase"
    )
    parser.add_argument(
        "--enable-disaggregation",
        action="store_true",
        help=(
            "Enable disaggregation mode (requires at least 2 GPUs): "
            "LLM on GPU 0, detokenizer on GPU 1 (default: False)"
        ),
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help=(
            "Enable data parallel mode with N replicas (default: 1, disables DP). "
            "Cannot be used with --enable-disaggregation. Requires N <= available GPUs."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--enable-nvtx", action="store_true", help="Enable NVTX profiling for performance analysis (default: False)"
    )
    parser.add_argument(
        "--enable-torch-compile",
        action="store_true",
        help="Enable torch.compile optimization for model inference (default: False)",
    )
    parser.add_argument(
        "--socket-suffix",
        type=str,
        default="",
        help="Suffix to append to IPC socket paths to avoid conflicts (default: empty)",
    )
    parser.add_argument(
        "--detokenize-interval",
        type=int,
        default=None,
        help="Interval for audio detokenization (default: None, model-specific). Only supported by qwen3-tts models.",
    )
    parser.add_argument(
        "--unroll-depth-cuda-graph",
        action="store_true",
        help="Unroll all depth transformer iterations into a single CUDA graph for reduced overhead (default: False)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("VOX_OUTPUT_DIR", "/tmp/vox_serve_audio"),
        help="Directory for uploads, registered voices, and non-streaming audio (default: /tmp/vox_serve_audio or VOX_OUTPUT_DIR)",
    )
    args = parser.parse_args()

    # Set global log level for the entire application
    set_global_log_level(args.log_level)

    # Set multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # Already set, ignore
        pass

    # Determine final CUDA graph setting
    enable_cuda_graph = args.enable_cuda_graph and not args.disable_cuda_graph

    # Validate data parallel mode
    if args.dp_size < 1:
        logger.error("--dp-size must be >= 1")
        sys.exit(1)

    # Check mutual exclusion between DP and disaggregation
    if args.dp_size > 1 and args.enable_disaggregation:
        logger.error(
            "Cannot enable both data parallel mode (--dp-size > 1) and disaggregation mode (--enable-disaggregation)"
        )
        logger.error("Please use one or the other")
        sys.exit(1)

    # Check GPU availability for data parallel
    if args.dp_size > 1:
        available_gpus = torch.cuda.device_count()
        if args.dp_size > available_gpus:
            logger.error(f"--dp-size {args.dp_size} exceeds available GPU count {available_gpus}")
            sys.exit(1)
        logger.info(f"Data parallel mode enabled with {args.dp_size} replicas (using GPUs 0-{args.dp_size - 1})")

    # Automatically select disaggregation scheduler if enable_disaggregation is set with CUDA graphs
    scheduler_type = args.scheduler_type
    if args.enable_disaggregation and enable_cuda_graph:
        logger.info(
            "Disaggregation mode enabled: using 'disaggregation' scheduler with parallel LM and detokenization loops"
        )
        scheduler_type = "disaggregation"

    # Construct socket paths with optional suffix
    request_socket_path = f"/tmp/vox_serve_request{args.socket_suffix}.ipc"
    result_socket_path = f"/tmp/vox_serve_result{args.socket_suffix}.ipc"

    # Initialize API server instance with specified model
    global api_server
    api_server = APIServer(
        model_name=args.model,
        scheduler_type=scheduler_type,
        request_socket_path=request_socket_path,
        result_socket_path=result_socket_path,
        output_dir=args.output_dir,
        max_batch_size=args.max_batch_size,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        cfg_scale=args.cfg_scale,
        greedy=args.greedy,
        enable_cuda_graph=enable_cuda_graph,
        enable_disaggregation=args.enable_disaggregation,
        enable_nvtx=args.enable_nvtx,
        enable_torch_compile=args.enable_torch_compile,
        async_scheduling=args.async_scheduling,
        dp_size=args.dp_size,
        detokenize_interval=args.detokenize_interval,
        unroll_depth_cuda_graph=args.unroll_depth_cuda_graph,
    )

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info(f"Starting vox-serve API server with model: {args.model}")
        logger.info("Scheduler and API server will be available shortly...")
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
    finally:
        if api_server is not None:
            api_server.cleanup()


if __name__ == "__main__":
    main()
