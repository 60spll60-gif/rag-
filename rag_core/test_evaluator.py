"""
核心逻辑单元测试 — pytest 运行

为什么写测试？
1. 重构时有安全网，改代码不怕引入新 bug
2. RAG 调优时，每次改动后跑测试就知道有没有退步

运行方式:
    pip install pytest pytest-cov
    pytest --cov=rag_core --cov-report=term-missing
"""
import pytest
from rag_core.evaluator import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_ndcg_at_k,
    evaluate_answer_faithfulness,
)
from rag_core.doc_utils import doc_key
from langchain_core.documents import Document


class TestRecallPrecision:
    """召回率 & 精确率测试"""

    def setup_method(self):
        """创建测试用文档"""
        self.docs = [
            Document(page_content="员工报销需要提交申请", metadata={"source": "hr_policy.pdf"}),
            Document(page_content="年假按工龄计算", metadata={"source": "hr_policy.pdf"}),
            Document(page_content="新员工培训为期两周", metadata={"source": "training.pdf"}),
        ]
        # 用 doc_key（来源+内容）作为稳定文档标识
        self.relevant_ids = {doc_key(self.docs[0]), doc_key(self.docs[1])}  # 2 个相关

    def test_recall_perfect(self):
        """所有相关文档都被召回 → recall=1.0"""
        result = compute_recall_at_k(self.docs[:2], self.relevant_ids, k=3)
        assert result == 1.0

    def test_recall_partial(self):
        """只召回了 1 个相关文档 → recall=0.5"""
        result = compute_recall_at_k([self.docs[0]], self.relevant_ids, k=3)
        assert result == 0.5

    def test_recall_zero(self):
        """没召回任何相关文档 → recall=0.0"""
        result = compute_recall_at_k([self.docs[2]], self.relevant_ids, k=3)
        assert result == 0.0

    def test_precision_perfect(self):
        """召回的全部都是相关的 → precision=1.0"""
        result = compute_precision_at_k(self.docs[:2], self.relevant_ids, k=3)
        assert result == 1.0

    def test_precision_mixed(self):
        """召回了不相关的 → precision < 1.0"""
        all_docs = self.docs + [Document(page_content="不相关的内容", metadata={})]
        result = compute_precision_at_k(all_docs[:3], self.relevant_ids, k=3)
        assert result < 1.0

    def test_empty_retrieved(self):
        """空结果 → 返回 0.0（不除零）"""
        assert compute_recall_at_k([], self.relevant_ids) == 0.0
        assert compute_precision_at_k([], self.relevant_ids) == 0.0

    def test_no_relevant_ids(self):
        """没有预期结果集合 → 返回 0.0"""
        assert compute_recall_at_k(self.docs[:2], set()) == 0.0
        assert compute_precision_at_k(self.docs[:2], set()) == 0.0

    def test_same_content_different_object_is_same_doc(self):
        """同一内容的不同对象实例应算同一个文档（doc_key 按内容去重）"""
        clone = Document(
            page_content="员工报销需要提交申请", metadata={"source": "hr_policy.pdf"}
        )
        result = compute_recall_at_k([clone], self.relevant_ids, k=3)
        assert result == 0.5  # 2 个相关文档召回了 1 个


class TestNDCG:
    """NDCG@K 排序质量测试"""

    def setup_method(self):
        self.docs = [
            Document(page_content="A", metadata={"source": "s1"}),
            Document(page_content="B", metadata={"source": "s2"}),
            Document(page_content="C", metadata={"source": "s3"}),
        ]
        # 理想排序：A 最相关(3)，B 次之(2)，C 最不相关(0)
        self.relevance = {
            doc_key(self.docs[0]): 3,
            doc_key(self.docs[1]): 2,
            doc_key(self.docs[2]): 0,
        }

    def test_perfect_order(self):
        """完美排序 → NDCG=1.0"""
        result = compute_ndcg_at_k(self.docs, self.relevance, k=3)
        assert result == pytest.approx(1.0)

    def test_bad_order(self):
        """相关度低的排前面 → NDCG < 1.0"""
        bad_order = [self.docs[2], self.docs[1], self.docs[0]]
        result = compute_ndcg_at_k(bad_order, self.relevance, k=3)
        assert result < 1.0

    def test_empty(self):
        """空输入 → 0.0"""
        assert compute_ndcg_at_k([], self.relevance) == 0.0
        assert compute_ndcg_at_k(self.docs, {}) == 0.0


class TestFaithfulness:
    """答案忠实度评估测试"""

    def test_high_overlap(self):
        """答案关键词与来源高度重叠 → faithfulness 高"""
        answer = "员工报销需要提交申请和审批两个步骤"
        docs = [Document(page_content="员工报销流程：先提交申请，再经过审批")]
        result = evaluate_answer_faithfulness(answer, docs)
        assert result["faithful"] is True
        assert result["coverage"] > 0

    def test_low_overlap(self):
        """答案关键词与来源完全不重叠 → faithfulness 低"""
        answer = "这个问题非常复杂，需要从多个维度来分析"
        docs = [Document(page_content="食堂午饭补贴是二十元每天")]
        result = evaluate_answer_faithfulness(answer, docs)
        assert not result["faithful"] or result["coverage"] < 0.3

    def test_empty_answer(self):
        """空答案 → coverage=0"""
        result = evaluate_answer_faithfulness("", [])
        assert result["coverage"] == 0.0

    def test_short_answer(self):
        """极短答案 → 无法有效评估"""
        result = evaluate_answer_faithfulness("对", [Document(page_content="是的")])
        # 太短的话 coverage 可能为 1.0（标记为难以评估）
        assert result["coverage"] >= 0
