import unittest
from fastapi.testclient import TestClient
import json

from sglang.srt.entrypoints.http_server import app, set_global_state, _GlobalState
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
import threading
from collections import defaultdict, deque
from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput
from sglang.srt.managers.template_manager import TemplateManager
from typing import List, Union

# Use a TokenizerManager subclass to get inline-state helpers
class DummyTM(TokenizerManager):
    def __init__(self):
        self.server_args = ServerArgs(
            model_path="dummy",
            inline_tool=True,
            tool_call_parser="pythonic",
        )
        # Set up inline-tool state
        self._inline_state: defaultdict = defaultdict(lambda: {"resume": deque(), "resp": deque()})
        self._inline_state_lock = threading.RLock()

    def create_abort_task(self, adapted_request):
        return None

    async def generate_request(self, adapted_request, request=None):
        rid = adapted_request.rid if isinstance(adapted_request.rid, str) else adapted_request.rid[0]
        yield {"meta_info": {"id": rid}, "text": "hello "}
        yield {"meta_info": {"id": rid}, "text": "[test_tool(message='foo')]"}
        yield {"meta_info": {"id": rid}, "text": " world"}

    async def pause_sequence(self, rid: str, reason: str = "tool"):
        # no-op stub
        return None

    async def enqueue_append_tokens(self, rid: str, token_ids: List[int]):
        # no-op stub
        return None

    async def resume_sequence(self, rid: str):
        # Pop and wake the next resume event
        evt = self._rpop(rid, "resume")
        if evt:
            evt.set()
        return None

    async def append_text_and_resume(self, rid: str, text: Union[str, List[int]], *, role=None):
        # Buffer tool response using real _renqueue
        self._renqueue(rid, "resp", (role or "tool", text))
        # Resume sequence stub
        await self.resume_sequence(rid)
        return True

class TestHTTPInlineToolResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Override the global state for HTTP server
        dummy_tm = DummyTM()
        set_global_state(_GlobalState(tokenizer_manager=dummy_tm,
                                      template_manager=TemplateManager(),
                                      scheduler_info={}))
        cls.client = TestClient(app)

    def test_resume_endpoint_and_streaming(self):
        payload = {
            "model": "dummy",
            "messages": [{"role": "user", "content": "foo"}],
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "test_tool", "parameters": {}}}],
        }
        # Initiate streaming chat completion
        with self.client.stream("POST", "/v1/chat/completions", json=payload) as response:
            rid = None
            # Read until we see the function_call chunk
            for line in response.iter_lines():
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                raw = line[len(b"data: "):]
                obj = json.loads(raw)
                choices = obj.get("choices", [])
                if choices and "function_call" in choices[0].get("delta", {}):
                    rid = obj.get("id")
                    break
            self.assertIsNotNone(rid, "Did not receive a function_call chunk to resume")

            # Call resume endpoint to send tool output
            resume_url = f"/v1/chat/completions/{rid}/resume"
            resp = self.client.post(resume_url, json={"text": "BAR"})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json().get("status"), "ok")

            # Continue reading stream to find the buffered response
            found = False
            for line in response.iter_lines():
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                raw = line[len(b"data: "):]
                obj = json.loads(raw)
                choices = obj.get("choices", [])
                content = choices[0].get("delta", {}).get("content")
                if content == "BAR":
                    found = True
                    break
            self.assertTrue(found, "Did not receive buffered tool response BAR in stream")

if __name__ == "__main__":
    unittest.main()