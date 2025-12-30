"""
Graph Builder.

Wires together the nodes into a compiled StateGraph.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.config import Settings
from app.graph.nodes.agent import agent_node
from app.graph.nodes.finalize import finalize_node
from app.graph.nodes.retrieval import retrieval_node
from app.graph.nodes.tools import tools_node
from app.graph.nodes.triage import triage_node
from app.graph.state import AgentState


def build_graph(settings: Settings):
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    # We use partials to inject 'settings' into nodes that need it
    workflow.add_node("triage", partial(triage_node, settings=settings))
    workflow.add_node("retrieval", partial(retrieval_node, settings=settings))
    workflow.add_node("agent", partial(agent_node, settings=settings))
    workflow.add_node("tools", partial(tools_node, settings=settings))
    workflow.add_node("finalize", partial(finalize_node, settings=settings))

    # 2. Define Edges

    # Start -> Triage
    workflow.set_entry_point("triage")

    # Triage -> Decision
    def route_triage(state: AgentState):
        if state["intent"] == "policy_query":
            return "retrieval"
        return "agent"

    workflow.add_conditional_edges(
        "triage",
        route_triage,
        {
            "retrieval": "retrieval",
            "agent": "agent"
        }
    )

    # Retrieval -> Agent
    workflow.add_edge("retrieval", "agent")

    # Agent -> Decision (Tools vs End)
    def route_agent(state: AgentState):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "finalize"

    workflow.add_conditional_edges(
        "agent",
        route_agent,
        {
            "tools": "tools",
            "finalize": "finalize"
        }
    )

    # Tools -> Agent (Loop back)
    workflow.add_edge("tools", "agent")

    # Finalize -> End
    workflow.add_edge("finalize", END)

    # 3. Compile
    # We pass checkpointer=None for now (in-memory only via script), 
    # but this is where Postgres checkpointer would go in production.
    return workflow.compile()