"""
统一配置中心 — 所有路径和参数集中管理，便于迁移和部署。
"""
import os
import sys
from pathlib import Path

# ── 基础路径 ───────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

# ── 数据文件路径 ───────────────────────────────────────
# LLM 增强特征数据（优先使用，来自 M3 实验产出）
FEATURES_CSV = DATA_DIR / "batch_1000_features.csv"
# 原始评论数据（回退数据源）
RAW_CSV = Path("D:/cxdownload/online_shopping_10_cats.csv")
# DuckDB 数据库文件（可选，用于演示只读连接模式）
DUCKDB_PATH = DATA_DIR / "analytics.db"

# ── 服务配置 ───────────────────────────────────────────
HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.getenv("DASHBOARD_PORT", "8000"))
FRONTEND_URL = f"http://{HOST}:{PORT}"

# ── LLM API 配置（硅基流动 / SiliconFlow） ─────────────
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
LLM_CONCURRENCY = 20

# ── 流式管道默认参数 ───────────────────────────────────
PIPELINE_QPS = 50
PIPELINE_QUEUE_LIMIT = 500
PIPELINE_CONSUMERS = 2
PIPELINE_BATCH_SIZE = 50
PIPELINE_BATCH_TIMEOUT = 0.5
PIPELINE_DURATION = 30

# ── 背压控制 ───────────────────────────────────────────
BACKPRESSURE_HIGH = 0.85
BACKPRESSURE_LOW = 0.30

# ── 模型文件路径（来自 M2 实验） ────────────────────────
MODEL_PATH_CANDIDATES = [
    BASE_DIR / "data" / "model.pkl",
    Path("../../test7/5913123045_陈弦_实验七/code/model.pkl"),
    Path("../test7/5913123045_陈弦_实验七/code/model.pkl"),
]

# ── 辅助函数 ───────────────────────────────────────────
def check_data_file(path: Path) -> bool:
    """检查数据文件是否存在"""
    return path.exists() and path.is_file()


def resolve_model_path() -> Path | None:
    """自动查找 ML 模型文件"""
    for cand in MODEL_PATH_CANDIDATES:
        p = cand if cand.is_absolute() else (BASE_DIR / cand).resolve()
        if p.exists():
            return p
    return None


def check_llm_api_key() -> dict:
    """
    检测 LLM API Key 配置状态。
    返回 {active: bool, reason: str}
    """
    if SILICONFLOW_API_KEY:
        return {"active": True, "reason": ""}
    return {
        "active": False,
        "reason": "API_KEY_MISSING — 请设置环境变量 SILICONFLOW_API_KEY 以启用 LLM 完整功能",
    }
