"""
MCP server exposing spec_agent.py's graph as callable tools.

Two tools, matching the two phases of the graph's execution:
  - start_spec_run(spec): runs generate_plan, pauses at human_review,
    returns the plan for the calling client (e.g. Claude Desktop) to show
    a human and get a decision.
  - submit_review(run_id, action, feedback): resumes the paused run
    with the human's decision. Loops back into generate_plan on reject,
    or proceeds to generate_code/validate_code on approve — same as
    running spec_agent.py directly, just driven over MCP instead of
    the __main__ block.

Run with: python3 mcp_server.py
Then point any MCP client (Claude Desktop config, Claude Code, etc.) at it.
"""

import uuid
from typing import Any, Dict

from mcp.server.mcpserver import MCPServer
from langgraph.types import Command

from spec_agent import app  # the compiled graph from spec_agent.py

mcp = MCPServer("spec-agent")

# Maps a short-lived MCP-facing run_id -> the underlying LangGraph thread_id.
# Kept separate from thread_id so we can validate/reject unknown run_ids
# without exposing internal LangGraph config directly to the MCP client.
_active_runs: Dict[str, str] = {}


def _initial_state(spec: str) -> Dict[str, Any]:
    return {
        "spec": spec,
        "plan": None,
        "feedback": None,
        "feedback_history": [],
        "revision_count": 0,
        "approved": False,
        "code": None,
        "code_revision_count": 0,
        "validation_error": None,
        "error": None,
    }


@mcp.tool()
def start_spec_run(spec: str) -> Dict[str, Any]:
    """Start a new spec-driven code generation run.

    Runs the plan-generation step and pauses for human review. The response
    contains a run_id — pass that to submit_review with the reviewer's
    decision to continue the run.

    Args:
        spec: The natural-language specification to plan and implement.
    """
    run_id = str(uuid.uuid4())
    thread_id = f"mcp-{run_id}"
    _active_runs[run_id] = thread_id

    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["spec-agent", "mcp"],
    }

    result = app.invoke(_initial_state(spec), config=config)

    if result.get("error"):
        return {"run_id": run_id, "status": "escalated", "error": result["error"]}

    interrupt_payload = result.get("__interrupt__")
    return {
        "run_id": run_id,
        "status": "awaiting_review",
        "plan": interrupt_payload[0].value if interrupt_payload else result.get("plan"),
    }


@mcp.tool()
def submit_review(run_id: str, action: str, feedback: str = "") -> Dict[str, Any]:
    """Submit a human review decision for a paused spec-agent run.

    Args:
        run_id: The run_id returned by start_spec_run.
        action: Either "approve" or "reject".
        feedback: Required if action is "reject" — explains what to fix.
    """
    if run_id not in _active_runs:
        return {"status": "error", "error": f"Unknown run_id: {run_id}"}
    if action not in ("approve", "reject"):
        return {"status": "error", "error": "action must be 'approve' or 'reject'"}

    thread_id = _active_runs[run_id]
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["spec-agent", "mcp"],
    }

    decision = {"action": action}
    if action == "reject":
        decision["feedback"] = feedback

    result = app.invoke(Command(resume=decision), config=config)

    if result.get("error"):
        del _active_runs[run_id]
        return {"run_id": run_id, "status": "escalated", "error": result["error"]}

    interrupt_payload = result.get("__interrupt__")
    if interrupt_payload:
        # Rejected and looped back to generate_plan — still awaiting review.
        return {
            "run_id": run_id,
            "status": "awaiting_review",
            "plan": interrupt_payload[0].value,
        }

    # No interrupt and no error means the graph reached END successfully.
    del _active_runs[run_id]
    return {
        "run_id": run_id,
        "status": "done",
        "code": result.get("code"),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")