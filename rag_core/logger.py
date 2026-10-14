"""
统一日志配置 — 替代散乱的 print
1. 区分信息级别（INFO/WARNING/ERROR），方便排查问题
2. 同时输出到控制台和文件，历史可追溯
"""
import logging
import sys
from pathlib import Path


def setup_logger(name: str = "rag_knowledge_base", log_file: str = None):
    """创建统一 logger
    
    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        # 已经配置过，直接返回（避免重复添加 handler）
        return logger
    
    logger.setLevel(logging.INFO)
    
    # ---- 格式（带时间戳、级别、模块名）----
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # ---- 控制台输出 ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ---- 文件输出 ----
    if log_file is None:
        log_file = "logs/app.log"
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# 便捷函数
info = lambda msg: logging.getLogger("rag_knowledge_base").info(msg)
warning = lambda msg: logging.getLogger("rag_knowledge_base").warning(msg)
error = lambda msg: logging.getLogger("rag_knowledge_base").error(msg)
