import asyncio
import time
from pathlib import Path
from typing import Any
import sys

ROOT_DIR = Path(__file__).resolve().parent
src_path = str(ROOT_DIR / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import simple_solution.agent.graph as simple_graph
import src.agent.graph as src_graph

app = FastAPI()

ROOT_DIR = Path(__file__).resolve().parent

class CompareRequest(BaseModel):
    query: str
    provider: str = "openai"

def run_baseline(query: str, provider: str) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = simple_graph.run_agent(query, provider=provider)
        duration = time.perf_counter() - start
        
        # Format tools
        tools = []
        for t in result.tool_calls:
            tools.append({
                "name": t.name if not isinstance(t, dict) else t.get("name", ""),
                "args": t.args if not isinstance(t, dict) else t.get("args", {})
            })
            
        return {
            "status": "success",
            "time": f"{duration:.2f}s",
            "answer": result.final_answer,
            "tools": tools
        }
    except Exception as e:
        duration = time.perf_counter() - start
        return {
            "status": "error",
            "time": f"{duration:.2f}s",
            "answer": f"Error: {str(e)}",
            "tools": []
        }

def run_src(query: str, provider: str) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = src_graph.run_agent(query, provider=provider)
        duration = time.perf_counter() - start
        
        # Format tools
        tools = []
        for t in result.tool_calls:
            tools.append({
                "name": t.name if not isinstance(t, dict) else t.get("name", ""),
                "args": t.args if not isinstance(t, dict) else t.get("args", {})
            })
            
        return {
            "status": "success",
            "time": f"{duration:.2f}s",
            "answer": result.final_answer,
            "tools": tools
        }
    except Exception as e:
        duration = time.perf_counter() - start
        return {
            "status": "error",
            "time": f"{duration:.2f}s",
            "answer": f"Error: {str(e)}",
            "tools": []
        }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = ROOT_DIR / "compare_models.html"
    return html_path.read_text(encoding="utf-8")

@app.post("/api/compare")
async def compare_agents(req: CompareRequest):
    # Run both agents concurrently using threads to avoid blocking the event loop
    baseline_task = asyncio.to_thread(run_baseline, req.query, req.provider)
    src_task = asyncio.to_thread(run_src, req.query, req.provider)
    
    baseline_result, src_result = await asyncio.gather(baseline_task, src_task)
    
    return {
        "baseline": baseline_result,
        "src": src_result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
