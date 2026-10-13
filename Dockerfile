# ============================================================
# RAG 企业知识库问答系统 — Dockerfile
# 多阶段构建，减小最终镜像体积
# ============================================================

# ---- 阶段 1: 编译/依赖安装 ----
FROM python:3.11-slim AS builder

WORKDIR /app

# 安装系统依赖（某些 Python 包需要编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# 先复制依赖文件 → 利用 Docker 缓存层
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 阶段 2: 最终运行镜像 ----
FROM python:3.11-slim

WORKDIR /app

# 从 builder 复制已安装的 Python 包
COPY --from=builder /usr/local/lib/python3.11/site-packages \
    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码
COPY rag_ingest/ ./rag_ingest/
COPY rag_core/ ./rag_core/
COPY rag_api/ ./rag_api/
COPY rag_ui/ ./rag_ui/
COPY config.yaml .
# 注意：不要 COPY .env —— API Key 一旦打进镜像层，任何人拿到镜像都能
# 用 docker history 读出来。密钥必须通过 docker run -e 在运行时注入。

# 创建日志目录
RUN mkdir -p logs

EXPOSE 8000

# 启动命令
CMD ["uvicorn", "rag_api.api:app", "--host", "0.0.0.0", "--port", "8000"]
