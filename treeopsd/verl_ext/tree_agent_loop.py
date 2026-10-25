"""`tree_tool_agent`: verl ToolAgentLoop that (a) records resumable snapshots after
every completed (action, observation) pair and (b) can resume from such a snapshot.

Snapshot t (1-based) is the exact state after observation o_t has been merged:
    n_ids       len(prompt_ids)     (the full token sequence so far: prompt + response)
    n_resp      len(response_mask)
    n_messages  len(messages)
    assistant_turns, user_turns
Everything a continuation needs is a PREFIX of the root's final token ids, response
mask/logprobs and message list, so snapshots are just offsets (cheap to ship through Ray).

A continuation request carries `tree_prefix` (non-tensor batch field) with the sliced
prefix; the loop then skips PENDING and starts in GENERATING, so the continuation is
sampled from exactly the root's token prefix (paper Eq. 3: S ~ P_old(. | H_{m,t})).
"""

from __future__ import annotations

import contextvars
from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

_PREFIX: contextvars.ContextVar = contextvars.ContextVar("treeopsd_tree_prefix", default=None)
_REC: contextvars.ContextVar = contextvars.ContextVar("treeopsd_tree_record", default=None)


@register("tree_tool_agent")
class TreeToolAgentLoop(ToolAgentLoop):
    """Per-request state lives in ContextVars: verl runs each sample as its own asyncio
    task, and each task gets its own copy of the context, so concurrent requests on the
    same loop instance cannot see each other's prefix or snapshots."""

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs):
        prefix = kwargs.pop("tree_prefix", None)
        kwargs.pop("tree_meta", None)
        rec = {"snapshots": list((prefix or {}).get("snapshots") or []), "messages": None}
        t1 = _PREFIX.set(prefix if prefix else None)
        t2 = _REC.set(rec)
        try:
            output = await super().run(sampling_params, priority=priority, **kwargs)
        finally:
            _PREFIX.reset(t1)
            _REC.reset(t2)
        # do NOT put these into agent_data.extra_fields during the loop: the parent merges
        # the engine's extra_fields only while that dict is still empty
        output.extra_fields["tree_snapshots"] = rec["snapshots"]
        output.extra_fields["tree_messages"] = [dict(m) for m in (rec["messages"] or [])]
        return output

    async def _handle_pending_state(self, agent_data, sampling_params):
        rec = _REC.get()
        prefix = _PREFIX.get()
        if not prefix:
            state = await super()._handle_pending_state(agent_data, sampling_params)
        else:
            agent_data.prompt_ids = list(prefix["prompt_ids"])
            agent_data.response_mask = list(prefix["response_mask"])
            agent_data.response_logprobs = list(prefix.get("response_logprobs") or [])
            agent_data.messages[:] = [dict(m) for m in prefix["messages"]]
            agent_data.assistant_turns = int(prefix["assistant_turns"])
            agent_data.user_turns = int(prefix["user_turns"])
            state = AgentState.GENERATING
        rec["messages"] = agent_data.messages  # same list object; grows with the episode
        return state

    async def _handle_processing_tools_state(self, agent_data):
        state = await super()._handle_processing_tools_state(agent_data)
        if state == AgentState.GENERATING:
            _REC.get()["snapshots"].append(
                {
                    "n_ids": len(agent_data.prompt_ids),
                    "n_resp": len(agent_data.response_mask),
                    "n_messages": len(agent_data.messages),
                    "assistant_turns": agent_data.assistant_turns,
                    "user_turns": agent_data.user_turns,
                }
            )
        return state
