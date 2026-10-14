"""
文档入库管线：加载 PDF/Word → 智能切分 → 输出文本块
支持从 YAML 配置读取所有参数
"""
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_core.config import load_config
from rag_core.logger import setup_logger, info, warning

logger = setup_logger()


def _sanitize_metadata(meta: dict) -> dict:
    """把 frontmatter 元数据转成向量库支持的类型

    YAML 会把 "2026-01-15" 解析成 date 对象、嵌套结构解析成 dict，
    而 ChromaDB 的元数据只接受 str/int/float/bool/list/None。
    """
    from datetime import date, datetime

    clean = {}
    for key, value in meta.items():
        if isinstance(value, (date, datetime)):
            clean[key] = value.isoformat()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        elif isinstance(value, list):
            clean[key] = [
                v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
                for v in value
            ]
        else:
            clean[key] = str(value)
    return clean


def _parse_frontmatter(filepath: Path) -> tuple:
    """解析 Markdown 的 YAML frontmatter，返回 (正文内容, 元数据字典)
    
    frontmatter 是文件开头被 --- 包裹的 YAML 块，例如：
        ---
        title: 网络故障处理
        category: 网络运维
        ---
        # 正文...
    
    解析后元数据会 attach 到每个 chunk，供后续：
    - 元数据过滤检索（按 category 限定范围）
    - 引用展示（显示标题/类别而非裸文件路径）
    - 增量索引（按文件哈希判断变化）
    """
    import yaml
    
    with open(filepath, "r", encoding="utf-8") as file:
        raw = file.read()
    
    meta = {"source": str(filepath), "title": filepath.stem}
    
    # 检测 frontmatter：文件以 --- 开头，且后面还有第二个 ---
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            try:
                fm = yaml.safe_load(parts[1]) or {}
                if isinstance(fm, dict):
                    meta.update(_sanitize_metadata(fm))
                content = parts[2].strip()
                return content, meta
            except yaml.YAMLError as e:
                logger.warning(f"{filepath} frontmatter 解析失败，按普通文本处理: {e}")
    
    return raw, meta


def load_documents(doc_dir: str = None) -> list:
    """从目录递归加载所有 PDF 和 Word 文档
    
    Args:
        doc_dir: 文档根目录路径（默认从 config.yaml 读取）
    
    Returns:
        加载后的 Document 列表
    """
    if doc_dir is None:
        config = load_config()
        doc_dir = config.get("ingest", {}).get("source_dir", "./test_docs")
    
    docs = []
    doc_path = Path(doc_dir)
    
    # PDF 文件加载
    pdf_files = sorted(doc_path.rglob("*.pdf"))
    for pdf_file in pdf_files:
        logger.info(f"[PDF] 正在加载: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs.extend(loader.load())
        except Exception as e:
            logger.warning(f"跳过文件 {pdf_file}: {e}")
    
    # Word 文件加载
    docx_files = sorted(doc_path.rglob("*.docx"))
    for docx_file in docx_files:
        logger.info(f"[DOCX] 正在加载: {docx_file}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            docs.extend(loader.load())
        except Exception as e:
            logger.warning(f"跳过文件 {docx_file}: {e}")

    # Markdown / TXT 文件加载（支持 YAML frontmatter 元数据）
    md_files = sorted(doc_path.rglob("*.md"))
    txt_files = sorted(doc_path.rglob("*.txt"))
    for f in md_files + txt_files:
        logger.info(f"[{'MD' if f.suffix == '.md' else 'TXT'}] 正在加载: {f}")
        try:
            content, meta = _parse_frontmatter(f)
            from langchain_core.documents import Document
            docs.append(
                Document(page_content=content, metadata=meta)
            )
        except Exception as e:
            logger.warning(f"跳过文件 {f}: {e}")
        
    if not docs:
        warning(f"在 {doc_dir} 中未找到 PDF、Word、Markdown 或 TXT 文档")
        warning("请将 .pdf / .docx / .md / .txt 文件放入该目录")
        
    return docs


def split_documents(docs: list) -> list:
    """将文档智能切分成小块
    
    关键参数说明：
    - chunk_size: 每块最大字符数。太小语义不完整，太大混入多个主题
    - chunk_overlap: 相邻块重叠字符数。防止一句话被切断在两半
    - separators: 中文优先分隔符（句号、逗号等），避免按空格乱切
    
    Args:
        docs: Document 列表
    
    Returns:
        切分后的文本块列表
    """
    config = load_config()
    ing_cfg = config.get("ingest", {})
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=ing_cfg.get("chunk_size", 500),
        chunk_overlap=ing_cfg.get("chunk_overlap", 80),
        separators=ing_cfg.get("separators", ["\n\n", "\n", "。", "；", "，", " ", ""]),
        length_function=len,
    )
    
    chunks = splitter.split_documents(docs)
    info(f"共切分为 {len(chunks)} 个文本块 (每块约 {ing_cfg.get('chunk_size', 500)} 字)")
    return chunks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="文档加载与切分")
    parser.add_argument("--doc-dir", default=None, help="文档目录（覆盖 config.yaml）")
    args = parser.parse_args()
    
    print(f"📂 正在扫描目录: {args.doc_dir or 'config.yaml 中的 source_dir'}")
    documents = load_documents(args.doc_dir)
    print(f"📄 共加载 {len(documents)} 个文档")
    
    chunks = split_documents(documents)
    print("\n--- 示例文本块 ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[块{i+1}] 来源: {chunk.metadata.get('source', 'unknown')}")
        print(f"内容: {chunk.page_content[:150]}...")
        
    print("\n💡 下一步: python -m rag_core.embed_and_store 进行向量化入库")
