# RAG 企业知识库智能问答系统

> 基于 **LangChain + BGE-M3 + ChromaDB + DeepSeek** 的企业级检索增强生成（RAG）系统  
> 覆盖完整链路：文档解析 → 智能切分 → 混合检索 → 重排序 → 答案生成

---

## 🏗️ 架构总览

```
                    ┌──────────────┐
                    │  用户输入     │
                    └──────┬───────┘
                           ▼
              ┌──────────────────────┐
              │   HTML / Streamlit    │  ← Web 前端 (rag_ui/)
              │   FastAPI Swagger     │  ← RESTful API (rag_api/)
              └──────────┬───────────┘
                         ▼
            ┌────────────────────────┐
            │    核心 RAG 引擎       │  ← rag_core/
            │                        │
            │  1. 向量粗排 (BGE-M3)  │  Dense Retrieval
            │  2. BM25 关键词召回    │  Sparse Retrieval
            │  3. RRF 融合排序       │  Hybrid Fusion
            │  4. CrossEncoder 精排  │  Re-Ranking
            │  5. LLM 生成答案       │  Generation
            └──────┬───────────┬────┘
                   ▼           ▼
          ┌──────────┐  ┌───────────┐
          │ ChromaDB │  │ DeepSeek  │
          │ 向量存储  │  │ LLM API  │
          └──────────┘  └───────────┘
                   
        ↑ 数据来源
    ┌──────────────┐
    │ PDF / Word   │  ← rag_ingest/
    │ Markdown     │    文档加载 + 智能切分
    └──────────────┘
```

## ✨ 核心特性

| 特性 | 说明 | 面试考点 |
|------|------|---------|
| **混合检索** | BM25(关键词) + Dense Vector(语义) + RRF融合 | 工业界标准检索方案 |
| **两段式排序** | 粗排快速召回 + CrossEncoder精细重排 | 召回质量提升的常见手段 |
| **配置驱动** | 所有超参集中 YAML 管理，不硬编码 | 工程规范性、实验可复现 |
| **结构化日志** | INFO/WARNING/ERROR 分级，文件+控制台双输出 | 生产环境排查问题必备 |
| **评估体系** | Recall@K, Precision@K, NDCG@K, Answer Faithfulness | "效果好"需要数字说话 |
| **容器化部署** | Dockerfile 多阶段构建，一键启动 | 生产部署能力 |
| **单元测试** | pytest + coverage，核心逻辑有测试保护 | 工程质量意识 |
| **RESTful API** | FastAPI + Pydantic 校验 + Swagger 文档 | 标准化接口设计 |
| **专业前端** | Bootstrap 5 + 原生 JS 聊天界面 | 全栈交付能力 |

## 🚀 快速开始

### Step 1: 安装依赖

```bash
pip install -r requirements.txt
```

首次运行会自动下载模型权重（约 3GB）：
- `BAAI/bge-m3` — 中文嵌入模型 ~2GB
- `BAAI/bge-reranker-v2-m3` — 重排序模型 ~1GB

### Step 2: 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
export OPENAI_API_KEY="sk-your-key"
export OPENAI_BASE_URL="https://api.deepseek.com"
```

### Step 3: 放入测试文档

将 PDF / Word 文件放入 `test_docs/` 目录

### Step 4: 执行完整管线

```bash
# ① 文档加载与切分（从 config.yaml 读取参数）
python -m rag_ingest.ingest --doc-dir ./test_docs

# ② 向量化入库（每次运行清空旧集合后重建，文档更新后重跑此命令即可）
python -m rag_core.embed_and_store

# ③ 启动 API 服务 + Web 前端
python -m rag_api.api

# ④ 浏览器打开 http://localhost:8000
```

### 可选：Docker 一键启动

```bash
docker build -t rag-kb .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key \
  -e OPENAI_BASE_URL=https://api.deepseek.com \
  -v $(pwd)/chroma_db:/app/chroma_db \
  rag-kb
```

> 注意：API Key 通过 `-e` 在运行时注入，不要打进镜像；`chroma_db/` 是数据目录，
> 用 `-v` 挂载，否则每次新建容器都是空知识库。

## 📊 关键参数调优指南

### chunk_size（每块大小）

| chunk_size | 效果 | 适用场景 |
|------------|------|---------|
| < 200 | 太碎，语义不完整 | ❌ 不推荐 |
| 400-600 | 语义完整且聚焦 | ✅ 中文文档推荐 |
| 800-1000 | 可能混入多个主题 | 长段落、报告类 |
| > 1500 | 什么都像什么都不像 | ❌ 避免 |

### chunk_overlap（重叠区域）

通常设为 chunk_size 的 10-20%，防止一句话被切断在两半。

### top_k_cascade vs top_k_final

- **top_k_cascade = 10**: 粗排先捞一批候选
- **top_k_final = 3**: 重排后给 LLM 的精选结果
- 原理：给重排模型留候选空间，否则粗排漏掉最佳答案就找不回来了

## 🧪 评估指标

```bash
# 运行单元测试
pip install pytest pytest-cov
pytest --cov=rag_core --cov-report=term-missing
```

评估维度：
- **Recall@K**: 相关文档中被成功召回的比例
- **Precision@K**: 召回的文档中有多少真正相关
- **NDCG@K**: 考虑相关程度和排名的排序质量
- **Answer Faithfulness**: 答案是否忠实于参考资料（有无幻觉）

使用方法：准备 20-50 个真实问题及每个问题的预期相关文档，调用
`RAGEvaluator.run(retrieve_fn, generate_fn)` 批量计算指标并保存报告。

## 💻 API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 前端页面 |
| `/health` | GET | 健康检查（liveness probe） |
| `/v1/ask` | POST | 智能问答 |

Swagger 文档：访问 `http://localhost:8000/docs`

示例请求：
```bash
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "报销流程是什么？", "top_k": 3}'
```

## 📁 项目结构

```
RAG-Enterprise-Knowledge/
├── config.yaml              # 全局配置（所有超参）
├── .env.example             # 环境变量模板
├── requirements.txt         # Python 依赖
├── pyproject.toml           # 项目元数据 + pytest 配置
├── Dockerfile               # 容器化部署
├── .dockerignore
├── README.md                # 本文件
│
├── rag_ingest/              # 📥 文档入库模块
│   ├── __init__.py
│   ├── ingest.py            # PDF/Word 加载 + 智能切分
│   └── test_ingest.py       # 单元测
│
├── rag_core/                # ⚙️ 核心逻辑包
│   ├── __init__.py
│   ├── config.py            # YAML 配置加载器
│   ├── logger.py            # 统一日志系统
│   ├── doc_utils.py         # 文档块稳定标识（融合/评估共用）
│   ├── embed_and_store.py   # 向量化 + ChromaDB（重建式入库）
│   ├── retrieval.py         # 检索 + 重排 + LLM 生成（主链路）
│   ├── hybrid_retrieval.py  # BM25 + 向量混合检索 + RRF 融合（已接入主链路）
│   ├── evaluator.py         # RAG 评估框架（Recall/Precision/NDCG/Faithfulness）
│   ├── test_evaluator.py    # 评估指标单元测试
│   └── test_hybrid.py       # RRF 融合单元测试
│
├── rag_api/                 # 🔌 生产接口
│   └── api.py               # FastAPI + 静态文件服务
│
├── rag_ui/                  # 🎨 前端界面
│   ├── index.html           # 双栏聊天界面（原生 JS：轻量 Markdown 渲染、引用溯源、复制答案）
│   ├── app.py               # Streamlit 版（备用）
│   └── assets/vendor/       # 本地化 CSS/字体（离线可用，不依赖 CDN）
│
├── test_docs/               # 📄 测试文档目录
│
└── logs/                    # 📝 日志输出（自动生成）
```

## 🎯 面试高频考点 & 本项目对应实践

| 面试问题 | 本项目做法 |
|----------|-----------|
| RAG vs 微调怎么选？ | RAG 适合知识频繁更新+需溯源；微调改变模型行为风格 |
| 怎么抑制幻觉？ | Prompt约束 + 低temperature(0.1) + 答案溯源标注 |
| 为什么用混合检索？ | 向量懂语义但漏精确匹配；BM25懂精确匹配但不懂语义 |
| RRF 融合怎么做？ | score(doc) = Σ 1/(k + rank)，k=60 经验值，两路按名次合并去重 |
| 重排序为什么必要？ | 粗排快但不够准；CrossEncoder 把问题和候选拼起来精打分 |
| 怎么评估 RAG？ | Recall@K, Precision@K, NDCG@K, Faithfulness，自建评估框架（RAGAS 思路） |
| 文档更新了怎么办？ | 重新执行入库命令，程序清空旧向量库后全量重建（按文件 MD5 增量更新是后续方向） |
| chunk_size 怎么选？ | 中文 400-600，太小语义不完整，太大混多个主题，需要在自己语料上实测 |
| 怎么优化检索质量？ | 混合检索 + 重排 + 调整 chunk 策略 + 用测试集量化验证 |

## ⚙️ 配置说明

所有可调参数集中在 `config.yaml`，修改无需改代码：

```yaml
ingest:
  chunk_size: 500       # 每块字符数上限
  chunk_overlap: 80     # 相邻块重叠

retrieval:
  top_k_cascade: 10     # 粗排候选数
  top_k_final: 3        # 重排后返回数
  enable_reranker: true # 启用 CrossEncoder 重排

llm:
  model: deepseek-chat
  temperature: 0.1      # 低温度求稳
```

## 🛤️ 后续演进方向

这些是面试官可能会建议的优化方向，你可以提前准备：

1. **增量索引**：按文件 MD5 判断是否变化，只重建变化的文档（当前为全量重建）
2. **层级索引**：先检索章节摘要，再下钻到具体块
3. **RAGAS 自动化评估**：构建测试集，自动化计算 Faithfulness/Context Precision
4. **多模态支持**：图片、表格解析（PDFplumber + 视觉模型）
5. **Agent 扩展**：将 RAG 作为工具接入 Agent，支持调用外部 API
6. **Milvus 替换 Chroma**：亿级向量分布式检索
7. **权限过滤**：按部门/角色在检索阶段过滤文档（企业知识库的核心诉求）
8. **流式返回**：SSE 边生成边推送，降低首字延迟
9. **依赖迁移**：langchain-community 已进入维护期，逐步迁移到独立集成包（langchain-chroma 等）
