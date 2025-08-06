import asyncio
import unittest

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase
from sglang.srt.entrypoints.openai.protocol import Tool

def make_test_tool():
    return [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {"message": {"type": "string"}}},
            },
        }
    ]

class TestOfflineInlineTool(CustomTestCase):
    def test_offline_inline_tool_sync_auto(self):
        """Verify automatic inline-tool parsing in synchronous generator."""
        server_args = ServerArgs(
            model_path="gpt2",
            skip_server_warmup=True,
            stream=True,
            grammar_backend="llguidance",
            inline_tool=True,
            tool_call_parser="pythonic",
        )
        engine = Engine(server_args=server_args)
        # Stub out RPC socket so inline-tool commands do not require a live scheduler
        class DummyRPC:
            def send_pyobj(self, obj): pass
            def recv_pyobj(self): return None
        engine.send_to_rpc = DummyRPC()
        rid = "sync_rid"
        async def dummy_gen(obj, _):
            yield {"meta_info": {"id": rid}, "text": "hello "}
            yield {"meta_info": {"id": rid}, "text": "[test_tool(message='foo')]"}
            yield {"meta_info": {"id": rid}, "text": " world"}
        engine.tokenizer_manager.generate_request = dummy_gen
        tools = [Tool(**t) for t in make_test_tool()]
        gen = engine.generate(prompt="", sampling_params={"max_new_tokens": 3}, stream=True, tools=tools)
        chunk1 = next(gen)
        self.assertEqual(chunk1["delta"]["content"], "hello ")
        chunk2 = next(gen)
        self.assertIn("function_call", chunk2["delta"])
        self.assertEqual(chunk2["meta_info"]["id"], rid)
        append_resp = engine.append_text_and_resume(rid, "BAR")
        self.assertTrue(append_resp is not None)
        chunk3 = next(gen)
        self.assertEqual(chunk3["delta"]["content"], "BAR")
        chunk4 = next(gen)
        self.assertEqual(chunk4["delta"]["content"], " world")
        with self.assertRaises(StopIteration):
            next(gen)
        engine.shutdown()

    def test_offline_inline_tool_async_auto(self):
        """Verify automatic inline-tool parsing in asynchronous generator."""
        server_args = ServerArgs(
            model_path="gpt2",
            skip_server_warmup=True,
            stream=True,
            grammar_backend="llguidance",
            inline_tool=True,
            tool_call_parser="pythonic",
        )
        engine = Engine(server_args=server_args)
        # Stub out RPC socket so inline-tool commands do not require a live scheduler
        class DummyRPC:
            def send_pyobj(self, obj): pass
            def recv_pyobj(self): return None
        engine.send_to_rpc = DummyRPC()
        rid = "async_rid"
        async def dummy_gen(obj, _):
            yield {"meta_info": {"id": rid}, "text": "hello "}
            yield {"meta_info": {"id": rid}, "text": "[test_tool(message='foo')]"}
            yield {"meta_info": {"id": rid}, "text": " world"}
        engine.tokenizer_manager.generate_request = dummy_gen
        tools = [Tool(**t) for t in make_test_tool()]

        async def run_async():
            gen = engine.async_generate(prompt="", sampling_params={"max_new_tokens": 3}, stream=True, tools=tools)
            chunk1 = await gen.__anext__()
            self.assertEqual(chunk1["delta"]["content"], "hello ")
            chunk2 = await gen.__anext__()
            self.assertIn("function_call", chunk2["delta"])
            self.assertEqual(chunk2["meta_info"]["id"], rid)
            append_resp = engine.append_text_and_resume(rid, "BAR")
            self.assertTrue(append_resp is not None)
            chunk3 = await gen.__anext__()
            self.assertEqual(chunk3["delta"]["content"], "BAR")
            chunk4 = await gen.__anext__()
            self.assertEqual(chunk4["delta"]["content"], " world")
            with self.assertRaises(StopAsyncIteration):
                await gen.__anext__()

        asyncio.run(run_async())
        engine.shutdown()