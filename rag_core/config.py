"""
配置加载器 — 从 YAML 文件读取全局配置

好处：
1. 实验对比时不用改代码，改配置就行
2. 不同环境用不同配置（开发/测试/生产）
3. 团队成员共享同一份配置模板
"""
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# 项目根目录（config.py 在 rag_core/ 下，所以 parent.parent 是根目录）
_PROJECT_ROOT = Path(__file__).parent.parent

# 模块导入时自动加载 .env（只需加载一次）
load_dotenv(_PROJECT_ROOT / ".env")


def load_config(config_path: str = None) -> dict:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径，默认在项目根目录找 config.yaml
    """
    if config_path is None:
        # 自动定位项目根目录的 config.yaml
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # ---- 环境变量覆盖（优先级最高）----
    # 这样 CI/CD 或生产环境可以直接设环境变量覆盖配置
    if "llm" in config:
        api_key_env = config["llm"].get("api_key_env", "OPENAI_API_KEY")
        config["llm"]["api_key"] = os.getenv(api_key_env, "")
        
        base_url_env = config["llm"].get("base_url_env", "OPENAI_BASE_URL")
        config["llm"]["base_url"] = os.getenv(base_url_env, "https://api.deepseek.com")
    
    return config


def get_llm_config() -> dict:
    """快捷获取 LLM 配置"""
    config = load_config()
    llm = config.get("llm", {})
    return {
        "api_key": llm.get("api_key", ""),
        "base_url": llm.get("base_url", "https://api.deepseek.com"),
        "model": llm.get("model", "deepseek-chat"),
        "temperature": llm.get("temperature", 0.1),
    }


def get_retrieval_config() -> dict:
    """快捷获取检索配置"""
    config = load_config()
    ret = config.get("retrieval", {})
    return {
        "top_k_cascade": ret.get("top_k_cascade", 10),
        "top_k_final": ret.get("top_k_final", 3),
        "enable_reranker": ret.get("enable_reranker", True),
        "enable_hybrid": ret.get("enable_hybrid", True),
    }
