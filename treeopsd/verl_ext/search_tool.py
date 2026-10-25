"""`search` tool (paper App. F.1): query_list of 1-4 queries (<= 512 chars each) ->
top-3 passages per query from the frozen retriever server (retriever/server.py).

The observation text follows Search-R1's format ("Doc i(Title: t) text"), one block per
query, and is truncated to `max_chars` (16,384 for MHQA, App. E.3).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import aiohttp

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


def format_hits(queries: List[str], results: List[List[Dict[str, Any]]], max_chars: int) -> str:
    blocks = []
    for q, hits in zip(queries, results):
        lines = [f"Query: {q}"]
        for i, h in enumerate(hits, 1):
            title = h.get("title", "")
            text = h.get("text", "")
            lines.append(f"Doc {i}(Title: {title}) {text}")
        blocks.append("\n".join(lines))
    out = "\n\n".join(blocks)
    return out[:max_chars]


def clean_queries(params: Dict[str, Any], max_queries: int = 4, max_len: int = 512) -> List[str]:
    q = params.get("query_list")
    if isinstance(q, str):
        q = [q]
    if not isinstance(q, list):
        return []
    q = [str(x).strip()[:max_len] for x in q if isinstance(x, (str, int, float)) and str(x).strip()]
    return q[:max_queries]


class FrozenSearchTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.url = os.environ.get("TREEOPSD_RETRIEVER_URL", config.get("retrieval_service_url"))
        self.topk = int(config.get("topk", 3))
        self.max_chars = int(config.get("max_chars", 16384))
        self.timeout = float(config.get("timeout", 120))

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        queries = clean_queries(parameters)
        if not queries:
            return ToolResponse(text="Error: query_list must be a non-empty JSON array of strings."), 0.0, {
                "num_queries": 0, "error": 1}
        payload = {"queries": queries, "topk": self.topk}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as s:
            async with s.post(self.url, json=payload) as r:
                r.raise_for_status()
                data = await r.json()
        return ToolResponse(text=format_hits(queries, data["results"], self.max_chars)), 0.0, {
            "num_queries": len(queries), "error": 0}
