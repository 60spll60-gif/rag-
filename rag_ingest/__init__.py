"""
rag_ingest — 文档入库模块
负责：加载 PDF/Word → 智能切分 → 输出文本块
"""
from rag_ingest.ingest import load_documents, split_documents

__all__ = ["load_documents", "split_documents"]
