"""文档块标识工具。

混合检索融合和评估都需要判断"两个文档是不是同一块"。
不能用 id(doc)：同一个文档块从 Chroma 取回和从 BM25 取回是两个不同的
Python 对象，id() 是内存地址，两者永远不相等，融合时会被当成两个文档。
"""


def doc_key(doc) -> tuple:
    """返回 (来源, 内容) 二元组，作为文档块的稳定标识。"""
    source = ""
    if doc.metadata:
        source = str(doc.metadata.get("source", ""))
    return (source, doc.page_content)
