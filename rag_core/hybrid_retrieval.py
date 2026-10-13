"""
混合检索模块：BM25(稀疏/关键词) + Dense Vector(稠密/语义) + RRF 融合 + 重排

为什么工业界要用混合检索？
1. 精确匹配场景：工号 E12345、产品型号 XYZ-900 → BM25 胜
2. 语义理解场景："报销能走电子流程吗" vs "电子发票报销" → 向量检索胜
3. 融合策略：两路各自召回 Top-K，按 RRF（Reciprocal Rank Fusion）合并排序
"""
from rag_core.config import load_config
from rag_core.logger import setup_logger
from rag_core.doc_utils import doc_key

logger = setup_logger()

# 进程内缓存 BM25 检索器，只在第一次使用时构建
_bm25_cache = {}


def get_bm25_retriever():
    """构建并缓存 BM25 检索器（关键词召回）

    从 config 指定的文档目录加载 + 切分，与向量库使用同一套切分参数，
    保证两路召回的是同一批文档块，融合时才能正确去重合并。
    """
    if "retriever" not in _bm25_cache:
        from langchain_community.retrievers import BM25Retriever
        from rag_ingest.ingest import load_documents, split_documents

        config = load_config()
        doc_dir = config.get("ingest", {}).get("source_dir", "./test_docs")

        docs = load_documents(doc_dir)
        chunks = split_documents(docs)

        retriever = BM25Retriever.from_documents(chunks)
        retriever.k = config.get("retrieval", {}).get("top_k_cascade", 10)
        _bm25_cache["retriever"] = retriever

        logger.info(f"BM25 索引构建完成: {len(chunks)} 个文档块 (来源: {doc_dir})")

    return _bm25_cache["retriever"]


def hybrid_search(dense_docs: list, bm25_docs: list, top_k: int = 10) -> list:
    """RRF 融合两路排序结果

    RRF 公式：score(doc) = Σ 1/(k + rank(doc))
    - k 通常取 60（经验值，避免极端排名影响过大）
    - 只比名次不比分数：两路检索的分数尺度不同，直接加权没有意义
    - 同一文档块在两路都出现时按 (来源, 内容) 去重后分数相加
    """
    k_rrf = 60

    scores = {}
    docs_map = {}

    for docs in (dense_docs, bm25_docs):
        for i, doc in enumerate(docs):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + i + 1)
            docs_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = [docs_map[key] for key, _ in ranked[:top_k]]

    logger.info(
        f"混合检索完成: 向量来源 {len(dense_docs)} 篇, BM25 来源 {len(bm25_docs)} 篇, "
        f"去重后 {len(ranked)} 篇, 返回 {len(result)} 篇"
    )
    return result


def hybrid_retrieve_and_rerank(query: str, top_k: int = None) -> list:
    """混合检索 + 重排序完整链路（主链路检索入口）

    1. 两路并行召回：BM25 关键词 + Dense Vector 语义
    2. RRF 融合：按名次合并两路结果
    3. CrossEncoder 重排：候选逐对精打分，确定最终顺序
    """
    config = load_config()
    ret_cfg = config.get("retrieval", {})

    if top_k is None:
        top_k = ret_cfg.get("top_k_final", 3)
    cascade_k = ret_cfg.get("top_k_cascade", 10)

    # 延迟导入，避免与 retrieval 模块形成循环依赖
    from rag_core.retrieval import _get_embedding_store, _get_reranker

    # Step 1: 两路召回
    store = _get_embedding_store()
    dense_retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": cascade_k},
    )
    dense_results = dense_retriever.invoke(query)
    bm25_results = get_bm25_retriever().invoke(query)

    logger.info(f"召回: 向量={len(dense_results)}, BM25={len(bm25_results)}")

    # Step 2: RRF 融合
    merged = hybrid_search(dense_results, bm25_results, top_k=cascade_k)

    if not merged:
        return []

    # Step 3: CrossEncoder 重排
    if ret_cfg.get("enable_reranker", True):
        reranker = _get_reranker()
        pairs = [[query, doc.page_content] for doc in merged]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(merged, scores), key=lambda x: x[1], reverse=True)
        result = [doc for doc, _ in ranked[:top_k]]
    else:
        result = merged[:top_k]

    logger.info(f"重排后返回 {len(result)} 个文档块")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("混合检索对比测试")
    print("=" * 60)

    from rag_core.retrieval import _get_embedding_store

    query = input("请输入问题: ") or "电脑无法上网怎么排查？"

    # 纯向量检索
    store = _get_embedding_store()
    dense_ret = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )
    dense_res = dense_ret.invoke(query)
    print(f"\n【向量检索】Top-3:")
    for i, d in enumerate(dense_res[:3]):
        print(f"  [{i+1}] {d.metadata.get('source')}")
        print(f"      {d.page_content[:80]}...")

    # 纯 BM25
    bm25_res = get_bm25_retriever().invoke(query)
    print(f"\n【BM25检索】Top-3:")
    for i, d in enumerate(bm25_res[:3]):
        print(f"  [{i+1}] {d.metadata.get('source')}")
        print(f"      {d.page_content[:80]}...")

    # 混合检索 + 重排
    hybrid_res = hybrid_retrieve_and_rerank(query, top_k=3)
    print(f"\n【混合检索+重排】Top-3:")
    for i, d in enumerate(hybrid_res[:3]):
        print(f"  [{i+1}] {d.metadata.get('source')} | {d.page_content[:80]}...")
