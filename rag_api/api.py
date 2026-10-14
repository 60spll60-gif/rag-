"""
FastAPI 生产接口：标准化 REST API + 静态文件服务

路由设计：
- GET  /          → 返回 Web 前端页面
- GET  /health    → 健康检查（K8s 探测用）
- POST /v1/ask   → 智能问答（核心接口）
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import time
from pathlib import Path

from rag_core.retrieval import ask as rag_ask
from rag_core.config import load_config

app = FastAPI(
    title="企业知识库问答 API",
    description="基于 RAG 的企业知识库智能问答系统 — FastAPI 生产级接口",
    version="1.0.0",
)

# ---- 挂载静态文件（HTML 前端 + 本地 CSS/字体）----
ui_dir = Path(__file__).parent.parent / "rag_ui"
app.mount("/assets", StaticFiles(directory=ui_dir / "assets"), name="assets")


@app.get("/")
def serve_frontend():
    """返回 Web 前端页面"""
    return FileResponse(ui_dir / "index.html")


@app.get("/health")
def health_check():
    """健康检查接口（liveness probe，外部监控据此判断服务是否存活）"""
    config = load_config()
    persist_dir = config.get("vectorstore", {}).get("persist_directory", "./chroma_db")
    return {
        "status": "ok",
        "version": "1.0.0",
        "vector_store": Path(persist_dir).exists(),
        "llm_configured": bool(config.get("llm", {}).get("api_key")),
    }


class QueryRequest(BaseModel):
    """问答请求体
    
    Pydantic 做输入校验，非法请求直接返回 422，不用在业务逻辑里写 if
    """
    question: str
    top_k: Optional[int] = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    """问答响应体"""
    answer: str
    sources: list
    elapsed_seconds: float


@app.post("/v1/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    """智能问答主接口
    
    Args:
        request: 包含问题和参数
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    start = time.time()
    try:
        result = rag_ask(request.question, request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")
    
    elapsed = round(time.time() - start, 2)
    
    return QueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        elapsed_seconds=elapsed,
    )


if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    api_cfg = cfg.get("api", {})
    uvicorn.run(app, host=api_cfg.get("host", "0.0.0.0"), port=api_cfg.get("port", 8000))
