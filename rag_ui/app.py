"""
Streamlit 对话界面：50 行出可交互 demo
"""
import streamlit as st
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag_core.retrieval import ask as rag_ask


st.set_page_config(page_title="企业知识库问答", page_icon="📚", layout="wide")
st.title("📚 企业知识库智能问答")
st.caption("基于 RAG 技术 — 检索增强生成，答案带溯源")

# 初始化会话历史
if "history" not in st.session_state:
    st.session_state.history = []
if "sources_history" not in st.session_state:
    st.session_state.sources_history = []


def format_sources(sources: list, raw_docs: list) -> str:
    """格式化来源信息"""
    if not sources:
        return ""
    
    result = ""
    for i, src in enumerate(sources):
        result += f"\n\n**📄 文档{i+1}**: {src.get('file', 'unknown')}"
    return result


# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    top_k = st.slider("返回参考文档数", min_value=1, max_value=10, value=3)
    if st.button("清空对话"):
        st.session_state.history = []
        st.session_state.sources_history = []
        st.rerun()
    st.divider()
    st.info("""
    **技术栈**: LangChain + BGE-M3 + ChromaDB + DeepSeek
    
    **面试考点**:
    - 文档切分策略
    - 向量检索 vs 关键词检索
    - 重排序优化
    - 幻觉抑制
    """)

# 显示历史消息
for i, (question, answer, sources) in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        st.write(answer)
        
        # 来源展开器
        if sources:
            with st.expander(f"📎 查看引用来源 ({len(sources)} 个文档块)", expanded=False):
                for j, src in enumerate(sources):
                    file_name = src.get("file", "unknown")
                    st.markdown(f"**📄 文档{j+1}**: `{file_name}`")
                    try:
                        doc = st.session_state.sources_history[i]["raw_docs"][j]
                        st.code(doc.page_content[:300], language=None)
                    except (IndexError, KeyError):
                        pass

# 输入框
question = st.chat_input("请输入你的问题…")
if question:
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索知识库…"):
            result = rag_ask(question)
        
        answer = result["answer"]
        st.write(answer)
        
        st.session_state.history.append((question, answer, []))
        st.session_state.sources_history.append(result)
        
        # 来源展示
        if result.get("sources"):
            with st.expander(f"📎 查看引用来源 ({len(result['sources'])} 个文档块)", expanded=False):
                for j, src in enumerate(result["sources"]):
                    file_name = src.get("file", "unknown")
                    st.markdown(f"**📄 文档{j+1}**: `{file_name}`")
                    try:
                        doc = result["raw_docs"][j]
                        st.code(doc.page_content[:300], language=None)
                    except (IndexError, KeyError):
                        pass
        
        st.session_state.history[-1] = (question, answer, result["sources"])
