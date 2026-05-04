from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.graph import build_graph
from app.state import create_initial_state

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Agentic Threat Intel API",
    description="AI-powered cybersecurity threat intelligence pipeline using RAG, Tavily web search, LangGraph, and local LLM reasoning.",
    version="1.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

graph = build_graph()


class AnalyzeRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Cybersecurity question to analyze.",
        examples=["Explain ransomware and include recent examples"],
    )
    include_trace: bool = Field(
        default=False,
        description="If true, returns execution trace for debugging/demo UI.",
    )


class AnalyzeResponse(BaseModel):
    query: str
    final_answer: str
    validation_report: dict[str, Any]
    topic: str | None = None
    route: str | None = None
    fast_path_used: bool | None = None
    web_results_count: int
    retrieved_docs_count: int
    runtime_seconds: float
    execution_trace: dict[str, Any] | None = None


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "agentic-threat-intel",
        "version": "1.1.0",
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start_time = time.perf_counter()

    try:
        state = create_initial_state(query)
        final_state = graph.invoke(state)

        runtime_seconds = round(time.perf_counter() - start_time, 2)

        execution_trace = final_state.get("execution_trace") or {}
        fast_path_used = bool(
            final_state.get("fast_path_used")
            or (
                isinstance(execution_trace, dict)
                and execution_trace
                .get("fast_path", {})
                .get("details", {})
                .get("decision") == "skip_critic_reviser"
            )
        )

        return AnalyzeResponse(
            query=query,
            final_answer=final_state.get("final_answer", ""),
            validation_report=final_state.get("validation_report", {}),
            topic=final_state.get("topic", ""),
            route=final_state.get("route", ""),
            fast_path_used=fast_path_used,
            web_results_count=len(final_state.get("web_results", []) or []),
            retrieved_docs_count=len(final_state.get("retrieved_docs", []) or []),
            runtime_seconds=runtime_seconds,
            execution_trace=dict(execution_trace) if request.include_trace else None,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Pipeline execution failed.",
                "message": str(exc),
            },
        ) from exc