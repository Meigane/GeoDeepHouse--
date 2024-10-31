# 日志工具
import logging

def setup_logger(name):
    logger = logging.getLogger(name)
    # 配置日志格式和级别
    return logger 