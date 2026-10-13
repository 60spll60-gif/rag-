"""
向量化 + 入库管线：文本块 → BGE-M3 编码 → ChromaDB 存储
支持从 YAML 配置读取所有参数
"""
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from rag_core.config import load_config
from rag_core.logger import setup_logger, info

logger = setup_logger()


def create_vectorstore(chunks: list, persist_directory: str = None) -> Chroma:
    """将切分好的文本块向量化并存入 ChromaDB
    
    Args:
        chunks: 切分后的文本块列表
        persist_directory: 向量库持久化路径（默认从 config.yaml 读取）
    
    Returns:
        持久化的向量数据库实例
    """
    if persist_directory is None:
        config = load_config()
        persist_directory = config.get("vectorstore", {}).get(
            "persist_directory", "./chroma_db"
        )
    
    emb_cfg = load_config().get("embeddings", {})
    
    logger.info(f"正在加载嵌入模型: {emb_cfg.get('model_name', 'BAAI/bge-m3')}")
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_cfg.get("model_name", "BAAI/bge-m3"),
        model_kwargs={"device": emb_cfg.get("device", "cpu")},
        encode_kwargs={"normalize_embeddings": emb_cfg.get("normalize_embeddings", True)},
    )
    
    logger.info(f"正在向量化并存储到 {persist_directory}...")

    # 先清空旧集合再重建：from_documents 是追加模式，
    # 不清理的话重复运行会把同一批文档存多份，检索结果出现重复
    try:
        Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        ).delete_collection()
        logger.info("已清空旧向量库，开始重建")
    except Exception:
        logger.info("无旧向量库，直接新建")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    vectorstore.persist()
    info(f"✅ 已向量化 {len(chunks)} 个文本块，保存在 {persist_directory}")
    return vectorstore


if __name__ == "__main__":
    from pathlib import Path
    from rag_ingest.ingest import load_documents, split_documents
    
    # 如果还没有向量库，自动执行完整入库流程
    if not Path("./chroma_db").exists():
        print("⚠️  还没有向量库，开始完整入库流程...")
        
        # Step 1: 加载文档
        documents = load_documents()
        print(f"📄 共加载 {len(documents)} 个文档")
        
        # Step 2: 智能切分
        chunks = split_documents(documents)
        
        # Step 3: 向量化入库
        create_vectorstore(chunks)
    else:
        print("✅ 向量库已存在，可直接使用")
