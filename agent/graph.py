from __future__ import annotations

from typing import Literal

from agent.nodes.diff import diff_node
from agent.nodes.embed import embed_node
from agent.nodes.fetch import fetch_node
from agent.nodes.parse import parse_node
from agent.nodes.persist import persist_node
from agent.nodes.reason import reason_node
from agent.state import AgentState
from langgraph.graph import END, StateGraph

NODE_FETCH: Literal["fetch_node"] = "fetch_node"
NODE_PARSE: Literal["parse_node"] = "parse_node"
NODE_EMBED: Literal["embed_node"] = "embed_node"
NODE_DIFF: Literal["diff_node"] = "diff_node"
NODE_REASON: Literal["reason_node"] = "reason_node"
NODE_PERSIST: Literal["persist_node"] = "persist_node"


def _route_after_fetch(state: AgentState) -> Literal["parse_node", "persist_node"]:
    return NODE_PARSE if state.get("new_filing_found") else NODE_PERSIST


def build_agent_graph():
    builder: StateGraph[AgentState] = StateGraph(AgentState)

    builder.add_node(NODE_FETCH, fetch_node)
    builder.add_node(NODE_PARSE, parse_node)
    builder.add_node(NODE_EMBED, embed_node)
    builder.add_node(NODE_DIFF, diff_node)
    builder.add_node(NODE_REASON, reason_node)
    builder.add_node(NODE_PERSIST, persist_node)

    builder.set_entry_point(NODE_FETCH)

    # Conditional edge after fetch_node
    builder.add_conditional_edges(
        NODE_FETCH,
        _route_after_fetch,
        {
            NODE_PARSE: NODE_PARSE,
            NODE_PERSIST: NODE_PERSIST,
        },
    )

    # Linear edges when new filing exists
    builder.add_edge(NODE_PARSE, NODE_EMBED)
    builder.add_edge(NODE_EMBED, NODE_DIFF)
    builder.add_edge(NODE_DIFF, NODE_REASON)
    builder.add_edge(NODE_REASON, NODE_PERSIST)

    builder.add_edge(NODE_PERSIST, END)

    return builder.compile()
