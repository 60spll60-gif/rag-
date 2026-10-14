"""
检索 + 重排 + 生成（核心链路）：用户问题 → 召回文档 → 重排序 → LLM 生成答案
"""
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

from rag_core.config import get_llm_config, get_retrieval_config, load_config
from rag_core.logger import setup_logger, info

logger = setup_logger()

# ---- 模型缓存（避免每次调用都重新加载）----
_model_cache = {}


def _get_embedding_store():
    """加载已持久化的向量库（进程内缓存，避免每次请求重新加载）"""
    if "store" not in _model_cache:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        config = load_config()
        emb_cfg = config.get("embeddings", {})
        vs_cfg = config.get("vectorstore", {})

        embeddings = HuggingFaceEmbeddings(
            model_name=emb_cfg.get("model_name", "BAAI/bge-m3"),
            model_kwargs={"device": emb_cfg.get("device", "cpu")},
            encode_kwargs={"normalize_embeddings": emb_cfg.get("normalize_embeddings", True)},
        )

        _model_cache["store"] = Chroma(
            persist_directory=vs_cfg.get("persist_directory", "./chroma_db"),
            embedding_function=embeddings,
        )
    return _model_cache["store"]


def _get_reranker():
    """加载 BGE-Reranker-v2-m3 重排序模型（首次运行下载约 1GB）"""
    if "reranker" not in _model_cache:
        config = load_config()
        model_name = config.get("retrieval", {}).get(
            "reranker_model", "BAAI/bge-reranker-v2-m3"
        )
        logger.info(f"正在加载重排序模型: {model_name}")
        _model_cache["reranker"] = CrossEncoder(model_name)
    return _model_cache["reranker"]


def retrieve(query: str, top_k: int = None) -> list:
    """向量检索（粗排）

    先捞 Top-K 候选而不是直接取最终数：给重排留候选空间。
    相似度：归一化向量用余弦相似度。
    """
    cfg = get_retrieval_config()

    store = _get_embedding_store()
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": cfg["top_k_cascade"]},
    )

    docs = retriever.invoke(query)
    info(f"粗排召回 {len(docs)} 个候选文档")

    if top_k:
        return docs[:top_k]
    return docs


def retrieve_and_rerank(query: str, top_k: int = None) -> list:
    """向量检索 + 重排序（纯向量路径，供对比实验用；主链路默认走混合检索）
    
    两段式架构：
    1. 向量粗排：快速从大量候选中捞出 Top-10（毫秒级）
    2. CrossEncoder 重排：候选逐对精打分，返回真正最相关的 Top-3
    """
    if top_k is None:
        cfg = get_retrieval_config()
        top_k = cfg["top_k_final"]
    
    # Step 1: 向量粗排
    candidates = retrieve(query)
    if not candidates:
        return []
    
    # Step 2: 重排序精排
    enable_reranker = load_config().get("retrieval", {}).get("enable_reranker", True)
    if enable_reranker:
        logger.info(f"正在对 {len(candidates)} 个候选进行重排序...")
        reranker = _get_reranker()
        
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = reranker.predict(pairs)
        
        # 按分数从高到低排序
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        result = [doc for doc, score in ranked[:top_k]]
    else:
        result = candidates[:top_k]
    
    info(f"重排完成，返回 {len(result)} 个文档块")
    return result


def generate_answer(question: str, sources: list) -> dict:
    """组装 Prompt 并通过 OpenAI 兼容接口生成答案
    
    Prompt 工程是抑制幻觉的第一道防线：
    1. 角色定义 → 让模型知道自己是"知识库助手"而非自由聊天
    2. 引用标注要求 → 倒逼模型贴着资料说话
    3. "不知道就说不知道" → 防止编造
    """
    llm_cfg = get_llm_config()
    
    if not llm_cfg["api_key"]:
        return {
            "answer": "⚠️ 未配置 LLM API Key\n请先设置环境变量 OPENAI_API_KEY。\n详见 README.md",
            "sources": [],
        }
    
    from openai import OpenAI
    client = OpenAI(api_key=llm_cfg["api_key"], base_url=llm_cfg["base_url"])
    
    # 组装参考资料
    source_info = []
    context_parts = []
    for i, doc in enumerate(sources):
        source_file = doc.metadata.get("source", "unknown")
        title = doc.metadata.get("title", "")
        context_parts.append(f"[文档{i+1}] {doc.page_content}\n来源: {source_file}")
        source_info.append({"file": source_file, "title": title})
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""你是企业知识库助手。仅根据以下参考资料回答问题：

要求：
1. 答案中标注引用来源，如 [文档1]
2. 资料中没有的内容，明确说"知识库中未找到相关信息"，不要编造
3. 回答简洁，分点呈现

参考资料：
{context}

问题：{question}"""

    resp = client.chat.completions.create(
        model=llm_cfg["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=llm_cfg["temperature"],  # 低温度求稳
    )
    
    answer = resp.choices[0].message.content
    
    return {
        "answer": answer,
        "sources": source_info,
        "raw_docs": sources,
    }


def ask(question: str, top_k: int = None) -> dict:
    """一键问答入口

    完整 RAG 链路：
    User Question → 混合检索(BM25+向量) → RRF 融合 → 重排 → Prompt 拼装 → LLM → Answer
    """
    logger.info(f"收到问题: {question}")

    cfg = get_retrieval_config()
    if cfg.get("enable_hybrid", True):
        # 延迟导入，避免与 hybrid_retrieval 模块形成循环依赖
        from rag_core.hybrid_retrieval import hybrid_retrieve_and_rerank
        sources = hybrid_retrieve_and_rerank(question, top_k)
    else:
        sources = retrieve_and_rerank(question, top_k)

    if not sources:
        logger.info("未检索到相关文档，直接返回（不调用 LLM）")
        return {
            "answer": "知识库中未找到相关信息。请检查文档是否已入库（python -m rag_core.embed_and_store）。",
            "sources": [],
        }

    result = generate_answer(question, sources)

    return result


if __name__ == "__main__":
    query = input("请输入你的问题: ") or "电脑无法上网怎么排查？"
    
    print("\n" + "=" * 60)
    print(f"🔍 检索阶段:")
    print("=" * 60)
    sources = retrieve_and_rerank(query)
    for i, s in enumerate(sources):
        print(f"\n[文档{i+1}] 来源: {s.metadata.get('source')}")
        print(f"内容预览: {s.page_content[:200]}...")
    
    print("\n" + "=" * 60)
    print(f"✨ 生成阶段:")
    print("=" * 60)
    result = generate_answer(query, sources)
    print(f"\n答案:\n{result['answer']}")
