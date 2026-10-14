"""
RAG 评估模块 — 量化问答系统质量

为什么需要评估？
- "效果好"是主观感受，需要数字说话
- 优化前 vs 优化后，必须有可复现的指标对比

评估维度（RAGAS 思路简化版）：
1. Recall@K: 相关文档中被成功召回的比例（漏检了多少）
2. Precision@K: 召回的文档中有多少真正相关（混入了多少噪声）
3. NDCG@K: 考虑相关程度和排名的排序质量
4. Answer Faithfulness: 生成答案是否忠实于参考资料（有没有幻觉/编造）
"""
from typing import List, Dict
import math
import json

from rag_core.config import load_config
from rag_core.doc_utils import doc_key


def compute_recall_at_k(retrieved_docs: list, relevant_doc_ids: set, k: int = None) -> float:
    """计算 Recall@K: 相关文档中被成功召回的比例

    Args:
        retrieved_docs: 实际召回的文档列表
        relevant_doc_ids: 相关文档的 doc_key 集合（见 rag_core/doc_utils.py）
        k: 只评估前 K 个召回结果
    """
    if not retrieved_docs or not relevant_doc_ids:
        return 0.0

    top_k = retrieved_docs[:k] if k else retrieved_docs
    hit_count = sum(1 for doc in top_k if doc_key(doc) in relevant_doc_ids)
    return hit_count / len(relevant_doc_ids)


def compute_precision_at_k(retrieved_docs: list, relevant_doc_ids: set, k: int = None) -> float:
    """计算 Precision@K: 召回的文档中有多少是真正相关的

    Args:
        retrieved_docs: 实际召回的文档列表
        relevant_doc_ids: 相关文档的 doc_key 集合
        k: 只评估前 K 个召回结果
    """
    if not retrieved_docs or not relevant_doc_ids:
        return 0.0

    top_k = retrieved_docs[:k] if k else retrieved_docs
    hit_count = sum(1 for doc in top_k if doc_key(doc) in relevant_doc_ids)
    return hit_count / len(top_k)


def compute_ndcg_at_k(retrieved_docs: list, relevance_scores: dict, k: int = None) -> float:
    """计算 NDCG@K: 考虑相关程度加权的位置敏感评价指标

    - DCG: 相关文档排得越靠前，得分越高
    - IDCG: 理想情况下（最相关的排最前）的 DCG，只取前 K 个相关文档
    - NDCG = DCG / IDCG，归一化到 [0,1]
    """
    if not retrieved_docs or not relevance_scores:
        return 0.0

    top_k = retrieved_docs[:k] if k else retrieved_docs

    dcg = 0.0
    for i, doc in enumerate(top_k):
        rel = relevance_scores.get(doc_key(doc), 0)
        if rel > 0:
            dcg += rel / math.log2(i + 2)  # i+2: 位置从 1 开始，i 从 0 开始

    # 理想排序：所有相关文档按相关度从高到低排，只取前 K 个
    sorted_rels = sorted(relevance_scores.values(), reverse=True)
    top_k = len(top_k)
    idcg = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(sorted_rels[:top_k])
    )

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_answer_faithfulness(answer: str, source_docs: list) -> dict:
    """简易答案忠实度评估

    检查答案中的每个主张是否都能在参考资料中找到依据。
    简化实现：检查关键词重叠度。

    进阶方案：用 LLM 自己做判断（让另一个模型当裁判）

    Returns:
        {"faithful": bool, "coverage": float, "details": str}
    """
    import re

    if not answer.strip() or not source_docs:
        return {"faithful": False, "coverage": 0.0, "details": "无有效内容"}

    # 中文没有空格分词，用"字符二元组(bigram)"衡量相似度
    # 例如 "报销" -> {"报销"}; "提交申请" -> {"提交","交申","申请"}
    def bigrams(text: str) -> set:
        # 去掉标点和空白，只留有效字符
        cleaned = re.sub(r'[，。、；：\s\[\]【】()（）]', '', text)
        return {cleaned[i:i+2] for i in range(len(cleaned) - 1)}

    answer_grams = bigrams(answer)
    source_text = " ".join(doc.page_content for doc in source_docs)
    source_grams = bigrams(source_text)

    if not answer_grams:
        return {"faithful": True, "coverage": 1.0, "details": "答案过于简短，难以评估"}

    overlap = answer_grams & source_grams
    coverage = len(overlap) / len(answer_grams)

    return {
        "faithful": coverage > 0.3,  # 30% 二元组能对上就算可信
        "coverage": round(coverage, 3),
        "details": f"答案二元组 {len(answer_grams)} 个，与来源重叠 {len(overlap)} 个",
    }


class RAGEvaluator:
    """RAG 系统综合评估器

    使用方式：
    1. 创建测试集（问题 + 预期来源文档）
    2. 对每个问题执行检索和生成
    3. 批量计算各项指标
    """

    def __init__(self, test_cases: List[Dict]):
        """
        Args:
            test_cases: 测试用例列表，每项包含:
                - "question": 测试问题
                - "expected_sources": 期望找到的相关文档集合（Document 对象列表，
                  内部会转成 doc_key 集合）
                - "ground_truth": 标准答案（可选，用于更精细评估）
        """
        self.test_cases = test_cases
        self.results = []

    def run(self, retrieve_fn, generate_fn) -> dict:
        """执行完整评估

        Args:
            retrieve_fn: 检索函数，返回文档列表
            generate_fn: 生成函数，返回 {"answer": str, "sources": list}

        Returns:
            评估报告字典
        """
        results = []

        for i, tc in enumerate(self.test_cases):
            print(f"[{i+1}/{len(self.test_cases)}] 评估: {tc['question'][:50]}...")

            retrieved = retrieve_fn(tc["question"])
            generated = generate_fn(tc["question"], retrieved)

            relevant_keys = {doc_key(d) for d in tc.get("expected_sources", [])}

            recall = compute_recall_at_k(retrieved, relevant_keys, k=3)
            precision = compute_precision_at_k(retrieved, relevant_keys, k=3)
            faithfulness = evaluate_answer_faithfulness(
                generated.get("answer", ""), retrieved
            )

            result = {
                "question": tc["question"],
                "recall@3": recall,
                "precision@3": precision,
                "faithfulness": faithfulness,
                "answer": generated.get("answer", ""),
                "retrieved_count": len(retrieved),
            }
            results.append(result)

            print(f"  Recall@3: {recall:.3f}, Precision@3: {precision:.3f}")
            print(f"  Faithfulness: {faithfulness['coverage']:.3f}")

        avg_recall = sum(r["recall@3"] for r in results) / len(results) if results else 0.0
        avg_precision = sum(r["precision@3"] for r in results) / len(results) if results else 0.0
        avg_faithfulness = sum(r["faithfulness"]["coverage"] for r in results) / len(results) if results else 0.0

        report = {
            "total_tests": len(self.test_cases),
            "avg_recall_at_3": round(avg_recall, 3),
            "avg_precision_at_3": round(avg_precision, 3),
            "avg_faithfulness": round(avg_faithfulness, 3),
            "per_test_results": results,
        }

        self.results = results
        return report

    def save_report(self, filepath: str = "evaluation_report.json"):
        """保存评估报告到 JSON"""
        if self.results:
            summary = {
                "avg_recall_at_3": round(sum(r["recall@3"] for r in self.results) / len(self.results), 3),
                "avg_precision_at_3": round(sum(r["precision@3"] for r in self.results) / len(self.results), 3),
                "avg_faithfulness": round(sum(r["faithfulness"]["coverage"] for r in self.results) / len(self.results), 3),
            }
        else:
            summary = {"avg_recall_at_3": 0.0, "avg_precision_at_3": 0.0, "avg_faithfulness": 0.0}

        report = {
            "config": load_config().get("retrieval", {}),
            "metrics_summary": summary,
            "details": self.results,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"评估报告已保存到: {filepath}")
        return report


if __name__ == "__main__":
    from langchain_core.documents import Document

    relevant = [
        Document(page_content="员工报销需要提交申请", metadata={"source": "hr_policy.md"}),
        Document(page_content="年假按工龄计算", metadata={"source": "hr_policy.md"}),
    ]
    retrieved = relevant + [
        Document(page_content="食堂补贴二十元", metadata={"source": "canteen.md"}),
    ]

    evaluator = RAGEvaluator([
        {
            "question": "员工报销流程是什么？",
            "expected_sources": relevant,
        },
    ])
    report = evaluator.run(
        retrieve_fn=lambda q: retrieved,
        generate_fn=lambda q, docs: {"answer": "报销需要提交申请，年假按工龄计算。", "sources": docs},
    )
    print(report["avg_recall_at_3"], report["avg_precision_at_3"])
    print("提示: 真实使用请准备 20-50 个问题及其预期相关文档，再调用 run()")
