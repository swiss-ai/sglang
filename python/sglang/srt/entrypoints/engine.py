# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
The entry point of inference server. (SRT = SGLang Runtime)

This file implements python APIs for the inference engine.
"""

import asyncio
import atexit
import dataclasses
import logging
import multiprocessing as mp
import os
import signal
import threading
from typing import AsyncIterator, DefaultDict, Dict, Iterator, List, Optional, Tuple, Union
from collections import deque, defaultdict

import zmq
import zmq.asyncio
from PIL.Image import Image

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import torch
import uvloop

from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    MultimodalDataInputFormat,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    MultiprocessingSerializer,
    assert_pkg_version,
    configure_logger,
    get_zmq_socket,
    is_cuda,
    kill_process_tree,
    launch_dummy_health_check_server,
    prepare_model_and_tokenizer,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.version import __version__

## Inline-tool unified async streamer + sync wrapper
class _InlineToolEvent:
    """Resume event that can be awaited in async code and set from any thread."""
    def __init__(self, timeout: float):
        self._timeout = timeout
        self._evt = threading.Event()
    def set(self):
        self._evt.set()
    async def wait(self):
        """Wait for the resume event, raising asyncio.TimeoutError on timeout."""
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(loop.run_in_executor(None, self._evt.wait), timeout=self._timeout)

async def _inline_tool_stream_async(engine, raw_async_iter, parser):
    """Single async generator to handle inline tool calls."""
    rid = None
    try:
        async for chunk in raw_async_iter:
            rid = chunk["meta_info"]["id"]
            delta = chunk.get("text", "")
            # Safe parse tool calls
            try:
                normal_text, calls = parser.parse_stream_chunk(delta)
            except Exception as e:
                # Emit parse error to tool via buffer+resume and continue
                await engine.append_text_and_resume(rid, f"parse error: {e}", role="tool")
                yield {"delta": {"role": "tool", "content": f"parse error: {e}"}, "meta_info": chunk["meta_info"]}
                continue
            if normal_text:
                yield {"delta": {"role": "assistant", "content": normal_text}, "meta_info": chunk["meta_info"]}
            for call in calls:
                evt = _InlineToolEvent(engine.server_args.tool_timeout)
                # enqueue per-rid resume event
                engine._renqueue(rid, "resume", evt)
                engine.pause_sequence(rid)
                yield {"delta": {"function_call": {"name": call.name, "arguments": call.parameters}}, "meta_info": chunk["meta_info"]}
                try:
                    await evt.wait()
                except asyncio.TimeoutError:
                    engine._rpop(rid, "resp")
                    await engine.append_text_and_resume(rid, f"tool timeout after {engine.server_args.tool_timeout}s", role="tool")
                    yield {"delta": {"role": "tool", "content": f"tool timeout after {engine.server_args.tool_timeout}s"}, "meta_info": chunk["meta_info"]}
                else:
                    role, content = engine._rpop(rid, "resp") or (None, None)
                    if role is not None:
                        yield {"delta": {"role": role, "content": content}, "meta_info": chunk["meta_info"]}
                    else:
                        await engine.append_text_and_resume(rid, "no tool response provided", role="tool")
                        yield {"delta": {"role": "tool", "content": "no tool response provided"}, "meta_info": chunk["meta_info"]}
    finally:
        if rid:
            engine._cleanup_inline_state(rid)

def _sync_from_async(async_iter):
    """Wrap an async iterator into a sync generator by driving it on the current event loop."""
    def sync_gen():
        loop = asyncio.get_event_loop()
        try:
            while True:
                yield loop.run_until_complete(async_iter.__anext__())
        except StopAsyncIteration:
            pass
        finally:
            close = getattr(async_iter, "aclose", None)
            if close:
                loop.run_until_complete(close())
    return sync_gen()


logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

_is_cuda = is_cuda()


class Engine(EngineBase):
    """
    The entry point to the inference engine.

    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager both run in the main process.
    2. Inter-process communication is done through ICP (each process uses a different port) via the ZMQ library.
    """

    def __init__(self, **kwargs):
        """
        The arguments of this function is the same as `sglang/srt/server_args.py::ServerArgs`.
        Please refer to `ServerArgs` for the documentation.
        """
        if "server_args" in kwargs:
            # Directly load server_args
            server_args = kwargs["server_args"]
        else:
            # Construct server_args from kwargs
            if "log_level" not in kwargs:
                # Do not print logs by default
                kwargs["log_level"] = "error"
            server_args = ServerArgs(**kwargs)

        # Shutdown the subprocesses automatically when the program exits
        atexit.register(self.shutdown)

        # Allocate ports for inter-process communications
        self.port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

        # Launch subprocesses
        tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses(
            server_args=server_args,
            port_args=self.port_args,
        )
        self.server_args = server_args
        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager
        self.scheduler_info = scheduler_info

        context = zmq.Context(2)
        self.send_to_rpc = get_zmq_socket(
            context, zmq.DEALER, self.port_args.rpc_ipc_name, True
        )
        # Lock to protect RPC socket send/recv across threads
        self._rpc_lock = threading.RLock()
        # Lock for inline-tool state access
        self._inline_state_lock = threading.RLock()
        # per-rid inline-tool queues: resume events and buffered responses
        self._inline_state: DefaultDict[str, Dict[str, deque]] = defaultdict(lambda: {
            "resume": deque(),
            "resp":   deque(),
        })

    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: bool = False,
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        data_parallel_rank: Optional[int] = None,
    ) -> Union[Dict, Iterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        if self.server_args.enable_dp_attention:
            if data_parallel_rank is None:
                logger.debug("data_parallel_rank not provided, using default dispatch")
            elif data_parallel_rank < 0:
                raise ValueError("data_parallel_rank must be non-negative")
            elif data_parallel_rank >= self.server_args.dp_size:
                raise ValueError(
                    f"data_parallel_rank must be less than dp_size: {self.server_args.dp_size}"
                )

        # Warn if inline tool pause/resume requested without streaming
        if self.server_args.inline_tool:
            if not stream:
                logger.warning("inline-tool ignored as sync stream=False")
            elif tools is None:
                logger.warning("inline-tool ignored as tools=None")
            elif self.server_args.tool_call_parser is None:
                logger.warning("inline-tool ignored as tool_call_parser=None")
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
            return_hidden_states=return_hidden_states,
            stream=stream,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            data_parallel_rank=data_parallel_rank,
        )
        raw_async = self.tokenizer_manager.generate_request(obj, None)
        if stream and self.server_args.inline_tool and tools is not None and self.server_args.tool_call_parser is not None:
            parser = FunctionCallParser(tools, self.server_args.tool_call_parser)
            return _sync_from_async(_inline_tool_stream_async(self, raw_async, parser))
        raw = _sync_from_async(raw_async)
        return raw if stream else next(raw)

    async def async_generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: bool = False,
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        data_parallel_rank: Optional[int] = None,
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """

        if self.server_args.enable_dp_attention:
            if data_parallel_rank is None:
                logger.debug("data_parallel_rank not provided, using default dispatch")
            elif data_parallel_rank < 0:
                raise ValueError("data_parallel_rank must be non-negative")
            elif data_parallel_rank >= self.server_args.dp_size:
                raise ValueError(
                    f"data_parallel_rank must be in range [0, {self.server_args.dp_size-1}]"
                )

        logger.info(f"data_parallel_rank: {data_parallel_rank}")
        # Warn if inline tool pause/resume requested without streaming
        if self.server_args.inline_tool:
            if not stream:
                logger.warning("inline-tool ignored as async stream=False")
            elif tools is None:
                logger.warning("inline-tool ignored as tools=None")
            elif self.server_args.tool_call_parser is None:
                logger.warning("inline-tool ignored as tool_call_parser=None")
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            return_hidden_states=return_hidden_states,
            stream=stream,
            custom_logit_processor=custom_logit_processor,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            data_parallel_rank=data_parallel_rank,
        )
        raw_async = self.tokenizer_manager.generate_request(obj, None)
        if stream and self.server_args.inline_tool and tools is not None and self.server_args.tool_call_parser is not None:
            parser = FunctionCallParser(tools, self.server_args.tool_call_parser)
            return _inline_tool_stream_async(self, raw_async, parser)
        return raw_async if stream else raw_async.__anext__()

    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
    ) -> Dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(
            text=prompt,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
        )
        loop = asyncio.get_event_loop()
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = loop.run_until_complete(generator.__anext__())
        return ret

    async def async_encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
        image_data: Optional[MultimodalDataInputFormat] = None,
        audio_data: Optional[MultimodalDataInputFormat] = None,
        video_data: Optional[MultimodalDataInputFormat] = None,
    ) -> Dict:
        """
        Asynchronous version of encode method.

        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(
            text=prompt,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)
        return await generator.__anext__()

    def rerank(
        self,
        prompt: Union[List[List[str]]],
    ) -> Dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(text=prompt, is_cross_encoder_request=True)
        loop = asyncio.get_event_loop()
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = loop.run_until_complete(generator.__anext__())
        return ret

    def shutdown(self):
        """Shutdown the engine"""
        # Signal any pending inline-tool resume events and clean up state to avoid hanging generators
        with self._inline_state_lock:
            for state_map in self._inline_state.values():
                for evt_queue in state_map.values():
                    for evt in evt_queue:
                        evt.set()
            self._cleanup_inline_state()
        # Shutdown child processes
        kill_process_tree(os.getpid(), include_parent=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    def flush_cache(self):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.tokenizer_manager.flush_cache())

    def start_profile(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.tokenizer_manager.start_profile())

    def stop_profile(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.tokenizer_manager.stop_profile())

    def start_expert_distribution_record(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.tokenizer_manager.start_expert_distribution_record()
        )

    def stop_expert_distribution_record(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.tokenizer_manager.stop_expert_distribution_record()
        )

    def dump_expert_distribution_record(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.tokenizer_manager.dump_expert_distribution_record()
        )

    def get_server_info(self):
        loop = asyncio.get_event_loop()
        internal_states = loop.run_until_complete(
            self.tokenizer_manager.get_internal_state()
        )
        return {
            **dataclasses.asdict(self.tokenizer_manager.server_args),
            **self.scheduler_info,
            "internal_states": internal_states,
            "version": __version__,
        }

    def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ):
        """Initialize parameter update group."""
        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.init_weights_update_group(obj, None)
        )

    def update_weights_from_distributed(
        self,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str = "weight_update_group",
        flush_cache: bool = True,
    ):
        """Update weights from distributed source."""
        obj = UpdateWeightsFromDistributedReqInput(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=flush_cache,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_distributed(obj, None)
        )

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update weights from distributed source. If there are going to be more updates, set `flush_cache` to be false
        to avoid duplicated cache cleaning operation."""
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(self.server_args.tp_size)
            ],
            load_format=load_format,
            flush_cache=flush_cache,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_tensor(obj, None)
        )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: Optional[str] = None,
    ):
        """Update the weights from disk inplace without re-launching the engine.

        This method allows updating the model weights from disk without restarting
        the engine. It can be used to load a different model or update weights with
        new training.
        """
        obj = UpdateWeightFromDiskReqInput(
            model_path=model_path,
            load_format=load_format,
        )

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_disk(obj, None)
        )

    def get_weights_by_name(self, name: str, truncate_size: int = 100):
        """Get weights by parameter name."""
        obj = GetWeightsByNameReqInput(name=name, truncate_size=truncate_size)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.get_weights_by_name(obj, None)
        )

    def load_lora_adapter(self, lora_name: str, lora_path: str):
        """Load a new LoRA adapter without re-launching the engine."""

        obj = LoadLoRAAdapterReqInput(
            lora_name=lora_name,
            lora_path=lora_path,
        )

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.load_lora_adapter(obj, None)
        )

    def unload_lora_adapter(self, lora_name: str):
        """Unload a LoRA adapter without re-launching the engine."""

        obj = UnloadLoRAAdapterReqInput(lora_name=lora_name)

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.unload_lora_adapter(obj, None)
        )

    def release_memory_occupation(self, tags: Optional[List[str]] = None):
        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.release_memory_occupation(obj, None)
        )

    def resume_memory_occupation(self, tags: Optional[List[str]] = None):
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.resume_memory_occupation(obj, None)
        )

    def pause_sequence(self, rid: str, reason: str = "tool"):  # inline tool control
        """Pause a running sequence in the scheduler."""
        from sglang.srt.managers.io_struct import RpcReqInput

        req = RpcReqInput(method="pause_sequence", parameters={"rid": rid, "reason": reason})
        # protect socket usage
        with self._rpc_lock:
            self.send_to_rpc.send_pyobj(req)
            resp = self.send_to_rpc.recv_pyobj()
        return resp

    def append_text_and_resume(self, rid: str, text: Union[str, List[int]], *, role: str = None):
        """Append token_ids (or raw text) and resume decode inline."""
        from sglang.srt.managers.io_struct import RpcReqInput

        # If text is raw string, encode it; else assume token list
        token_ids = (
            text if isinstance(text, list) else self.tokenizer_manager.tokenizer.encode(text)
        )
        # record tool response content and buffer it
        resp_txt = text if isinstance(text, str) else self.tokenizer_manager.tokenizer.decode(text)
        # buffer role and content for both sync and async
        role_to_use = role or "tool"
        # buffer one response for the next resume
        self._renqueue(rid, "resp", (role_to_use, resp_txt))
        # Enqueue append
        enqueue_req = RpcReqInput(method="enqueue_append_tokens", parameters={"rid": rid, "token_ids": token_ids})
        with self._rpc_lock:
            self.send_to_rpc.send_pyobj(enqueue_req)
            _ = self.send_to_rpc.recv_pyobj()
        # Resume
        resume_req = RpcReqInput(method="resume_sequence", parameters={"rid": rid})
        with self._rpc_lock:
            self.send_to_rpc.send_pyobj(resume_req)
            _ = self.send_to_rpc.recv_pyobj()
        # wake the next resume event
        if evt := self._rpop(rid, "resume"):
            evt.set()
        return True

    def _cleanup_inline_state(self, rid: Optional[str] = None):
        """Clear inline-tool state(s)."""
        with self._inline_state_lock:
            if rid:
                self._inline_state.pop(rid, None)
            else:
                self._inline_state.clear()

    def _renqueue(self, rid: str, name: str, item) -> None:
        """Append an item to the per-rid `<name>` queue."""
        with self._inline_state_lock:
            self._inline_state[rid][name].append(item)

    def _rpop(self, rid: str, name: str):
        """Pop and return next item from the per-rid `<name>` queue, or None."""
        with self._inline_state_lock:
            q = self._inline_state[rid][name]
            if q:
                elem = q.popleft()
                # clean up empty rid-entry
                if not any(self._inline_state[rid].values()):
                    del self._inline_state[rid]
                return elem
        return None

    """
    Execute an RPC call on all scheduler processes.
    """

    def collective_rpc(self, method: str, **kwargs):
        obj = RpcReqInput(method=method, parameters=kwargs)
        self.send_to_rpc.send_pyobj(obj)
        recv_req = self.send_to_rpc.recv_pyobj(zmq.BLOCKY)
        assert isinstance(recv_req, RpcReqOutput)
        assert recv_req.success, recv_req.message

    def save_remote_model(self, **kwargs):
        self.collective_rpc("save_remote_model", **kwargs)

    def save_sharded_model(self, **kwargs):
        self.collective_rpc("save_sharded_model", **kwargs)

    def score(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> List[List[float]]:
        """
        Score the probability of specified token IDs appearing after the given (query + item) pair. For example:
        query = "<|user|>Is the following city the capital of France? "
        items = ["Paris <|assistant|>", "London <|assistant|>", "Berlin <|assistant|>"]
        label_token_ids = [2332, 1223] # Token IDs for "Yes" and "No"
        item_first = False

        This would pass the following prompts to the model:
        "<|user|>Is the following city the capital of France? Paris <|assistant|>"
        "<|user|>Is the following city the capital of France? London <|assistant|>"
        "<|user|>Is the following city the capital of France? Berlin <|assistant|>"
        The api would then return the probabilities of the model producing "Yes" and "No" as the next token.
        The output would look like:
        [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]]


        Args:
            query: The query text or pre-tokenized query token IDs. Must be provided.
            items: The item text(s) or pre-tokenized item token IDs. Must be provided.
            label_token_ids: List of token IDs to compute probabilities for. If None, no token probabilities will be computed.
            apply_softmax: Whether to normalize probabilities using softmax.
            item_first: If True, prepend items to query. Otherwise append items to query.

        Returns:
            List of dictionaries mapping token IDs to their probabilities for each item.
            Each dictionary in the list corresponds to one item input.

        Raises:
            ValueError: If query is not provided, or if items is not provided,
                      or if token IDs are out of vocabulary, or if logprobs are not available for the specified tokens.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.score_request(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=item_first,
                request=None,
            )
        )

    async def async_score(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> List[List[float]]:
        """
        Asynchronous version of score method.

        See score() for detailed documentation.
        """
        return await self.tokenizer_manager.score_request(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
            item_first=item_first,
            request=None,
        )


def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = str(int(server_args.enable_symm_mem))
    if not server_args.enable_symm_mem:
        os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.10",
            "Please uninstall the old version and "
            "reinstall the latest version by following the instructions "
            "at https://docs.flashinfer.ai/installation.html.",
        )
    if _is_cuda:
        assert_pkg_version(
            "sgl-kernel",
            "0.3.2",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    if True:  # Keep this check for internal code compatibility
        # Register the signal handler.
        # The child processes will send SIGQUIT to this process when any error happens
        # This process then clean up the whole process tree
        # Note: This sigquit handler is used in the launch phase, and may be replaced by
        # the running_phase_sigquit_handler in the tokenizer manager after the grpc server is launched.
        def launch_phase_sigquit_handler(signum, frame):
            logger.error(
                "Received sigquit from a child process. It usually means the child failed."
            )
            kill_process_tree(os.getpid())

        signal.signal(signal.SIGQUIT, launch_phase_sigquit_handler)

    # Set mp start method
    mp.set_start_method("spawn", force=True)


def _launch_subprocesses(
    server_args: ServerArgs, port_args: Optional[PortArgs] = None
) -> Tuple[TokenizerManager, TemplateManager, Dict]:
    """
    Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
    """
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []
    if server_args.dp_size == 1:
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        scheduler_pipe_readers = []

        nnodes_per_tp_group = max(server_args.nnodes // server_args.pp_size, 1)
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group),
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group + 1),
        )

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
                proc = mp.Process(
                    target=run_scheduler_process,
                    args=(
                        server_args,
                        port_args,
                        gpu_id,
                        tp_rank,
                        moe_ep_rank,
                        pp_rank,
                        None,
                        writer,
                        None,
                    ),
                )

                with memory_saver_adapter.configure_subprocess():
                    proc.start()
                scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()
        scheduler_procs.append(proc)

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        for reader in scheduler_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None, None

        launch_dummy_health_check_server(
            server_args.host, server_args.port, server_args.enable_metrics
        )

        for proc in scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
        return None, None, None

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(
            server_args,
            port_args,
        ),
    )
    detoken_proc.start()

    # Launch tokenizer process
    tokenizer_manager = TokenizerManager(server_args, port_args)

    # Initialize templates
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )

    # Wait for the model to finish loading
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            scheduler_procs[i].join()
            logger.error(f"Exit code: {scheduler_procs[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)

    # Assume all schedulers have the same scheduler_info
    scheduler_info = scheduler_infos[0]
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    return tokenizer_manager, template_manager, scheduler_info
