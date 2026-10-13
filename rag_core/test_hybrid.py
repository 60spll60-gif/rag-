"""
混合检索融合逻辑单元测试（不加载模型，纯函数测试）

重点验证 RRF 融合：
1. 同一文档块在两路都出现 → 去重合并，分数相加
2. 融合后的排序符合 RRF 公式预期
"""
import pytest
from langchain_core.documents import Document

from rag_core.hybrid_retrieval import hybrid_search
from rag_core.doc_utils import doc_key


def _doc(text, source="s.md"):
    return Document(page_content=text, metadata={"source": source})


class TestRRFFusion:
    """RRF 融合测试"""

    def test_same_doc_in_both_lists_merges(self):
        """同一内容在两路各出现一次 → 结果只保留一份，且分数相加后应排最前"""
        # 同一内容，两个不同对象实例（模拟 Chroma 和 BM25 各返回一份）
        doc_a_dense = _doc("VPN 连不上的排查步骤")
        doc_a_bm25 = _doc("VPN 连不上的排查步骤")
        doc_b = _doc("打印机墨盒更换方法")
        doc_c = _doc("新员工入职流程")

        dense = [doc_a_dense, doc_b]
        bm25 = [doc_a_bm25, doc_c]

        result = hybrid_search(dense, bm25, top_k=3)

        # 去重后应该是 3 个不同内容
        keys = [doc_key(d) for d in result]
        assert len(keys) == len(set(keys)) == 3
        # 双路都命中的文档应排第一（1/61 + 1/61 > 1/61）
        assert keys[0] == doc_key(doc_a_dense)

    def test_top_k_limits_result(self):
        """返回数量不超过 top_k"""
        dense = [_doc(f"文档{i}") for i in range(5)]
        bm25 = [_doc(f"另一批{i}") for i in range(5)]
        result = hybrid_search(dense, bm25, top_k=3)
        assert len(result) == 3

    def test_rank_order_follows_rrf(self):
        """单路内的相对排名应保持：靠前的名次贡献更大分数"""
        doc_first = _doc("排第一的内容")
        doc_second = _doc("排第二的内容")
        doc_third = _doc("排第三的内容")

        dense = [doc_second, doc_third]
        bm25 = [doc_first, doc_second]

        result = hybrid_search(dense, bm25, top_k=3)

        # doc_second 两路都有：1/61 + 1/62
        # doc_first: 1/61，doc_third: 1/62 → doc_first 应排第二
        keys = [doc_key(d) for d in result]
        assert keys[0] == doc_key(doc_second)
        assert keys[1] == doc_key(doc_first)
        assert keys[2] == doc_key(doc_third)

    def test_empty_both(self):
        """两路都为空 → 空结果"""
        assert hybrid_search([], [], top_k=3) == []

    def test_one_side_empty(self):
        """只有一路有结果 → 正常返回该路结果"""
        docs = [_doc(f"文档{i}") for i in range(3)]
        result = hybrid_search(docs, [], top_k=2)
        assert len(result) == 2
        assert result[0].page_content == "文档0"
