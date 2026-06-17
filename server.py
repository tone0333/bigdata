"""
FastAPI 数据看板服务 — 实验十四系统联调增强版。

特性：
- DuckDB 只读连接加载数据，避免与流式写入进程产生写锁冲突
- /api/system-status 端点：透传 LLM 降级状态供前端展示
- 数据文件缺失时自动回退降级，控制台输出醒目警告
- 所有 API 带防御性 try-except，不因单次查询崩溃
"""
import logging
import re
import sys
import traceback
from pathlib import Path

import duckdb
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── 把项目根目录加入路径，确保 config 可导入 ──────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

# ── 日志配置 ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dashboard")

# ── 系统状态（模块级，启动时检测） ─────────────────────
SYSTEM_STATUS = {
    "data_source": "unknown",
    "data_rows": 0,
    "llm_active": False,
    "llm_reason": "",
    "warnings": [],
}


def _color_warn(msg: str) -> str:
    """终端黄色警告文本（ANSI 转义）"""
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    return f"{YELLOW}{msg}{RESET}"


def load_data_with_duckdb():
    """
    使用 DuckDB 加载数据，支持只读连接模式。

    数据源优先级（自动降级）：
      1. DuckDB 数据库文件 (data/analytics.db) — 使用 read_only=True，防止写锁冲突
      2. LLM 增强特征 CSV — 使用 DuckDB CSV 查询
      3. 原始评论 CSV — 使用 DuckDB CSV 查询
      4. pandas 直接读取 — 最后回退
      5. 空 DataFrame — 不崩溃

    若连接的是持久化 DuckDB 文件，强制使用 read_only=True，
    避免与流式写入 Worker 产生写锁冲突。
    """
    global SYSTEM_STATUS

    csv_to_read: Path | None = None
    csv_label: str = ""

    if config.check_data_file(config.FEATURES_CSV):
        csv_to_read = config.FEATURES_CSV
        csv_label = "llm_features"
    elif config.RAW_CSV.exists():
        csv_to_read = config.RAW_CSV
        csv_label = "raw_csv"

    if csv_to_read is None:
        msg = "所有数据源均不可用！请将 batch_1000_features.csv 放入 data/ 目录"
        logger.warning(_color_warn(msg))
        SYSTEM_STATUS["data_source"] = "none"
        SYSTEM_STATUS["warnings"].append(msg)
        return pd.DataFrame(
            columns=["cat", "sentiment", "category", "review", "summary", "label"]
        )

    # ── 策略 1：如果存在 DuckDB 持久化数据库，使用 read_only=True ──
    if config.check_data_file(config.DUCKDB_PATH):
        logger.info("数据源: DuckDB 数据库 (%s) [read_only=True]", config.DUCKDB_PATH)
        SYSTEM_STATUS["data_source"] = "duckdb"
        try:
            conn = duckdb.connect(
                database=str(config.DUCKDB_PATH), read_only=True
            )
            table_name = "features" if csv_label == "llm_features" else "reviews"
            result = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
            conn.close()
            if len(result) > 0:
                SYSTEM_STATUS["data_rows"] = len(result)
                return result
            logger.warning("DuckDB 数据库为空，尝试从 CSV 加载")
        except Exception as e:
            msg = f"DuckDB 数据库读取失败: {e}"
            logger.warning(_color_warn(msg))
            SYSTEM_STATUS["warnings"].append(msg)

    # ── 策略 2：使用 DuckDB 查询 CSV 文件（无需 read_only，CSV 无写锁风险） ──
    logger.info("数据源: %s (%s)", csv_label, csv_to_read.name)
    SYSTEM_STATUS["data_source"] = csv_label
    try:
        conn = duckdb.connect(":memory:")
        conn.execute(f"CREATE TABLE tmp AS SELECT * FROM read_csv_auto('{csv_to_read.as_posix()}')")
        result = conn.execute("SELECT * FROM tmp").fetchdf()
        conn.close()

        # 兼容：确保有 sentiment 列
        if "sentiment" not in result.columns or result["sentiment"].isna().all():
            if "label" in result.columns:
                logger.info("从 label 列映射 sentiment")
                result["sentiment"] = result["label"].map(
                    {1: "正面", 0: "负面"}
                ).fillna("未知")
            else:
                result["sentiment"] = "未知"

        SYSTEM_STATUS["data_rows"] = len(result)
        return result

    except Exception as e:
        msg = f"DuckDB 读取 CSV 失败: {e}，回退到 pandas"
        logger.warning(_color_warn(msg))
        SYSTEM_STATUS["warnings"].append(msg)

    # ── 策略 3：pandas 直接读取（最后回退） ──
    try:
        result = pd.read_csv(csv_to_read, encoding="utf-8-sig")
        logger.info("通过 pandas 加载 %d 条记录", len(result))
        if "sentiment" not in result.columns and "label" in result.columns:
            result["sentiment"] = result["label"].map({1: "正面", 0: "负面"}).fillna("未知")
        SYSTEM_STATUS["data_rows"] = len(result)
        return result
    except Exception as e:
        msg = f"pandas 读取也失败: {e}"
        logger.warning(_color_warn(msg))
        SYSTEM_STATUS["warnings"].append(msg)

    # ── 最后防线 ──
    msg = "所有加载策略均失败！API 将以空数据运行"
    logger.warning(_color_warn(msg))
    SYSTEM_STATUS["data_source"] = "none"
    SYSTEM_STATUS["warnings"].append(msg)
    return pd.DataFrame(
        columns=["cat", "sentiment", "category", "review", "summary", "label"]
    )


# ── LLM 状态检测 ───────────────────────────────────────
llm_status = config.check_llm_api_key()
SYSTEM_STATUS["llm_active"] = llm_status["active"]
SYSTEM_STATUS["llm_reason"] = llm_status["reason"]
if not llm_status["active"]:
    logger.warning(_color_warn(llm_status["reason"]))
    logger.warning(_color_warn("大模型功能已降级 — 情感/类别数据依赖预计算特征或规则库"))

# ── 加载数据 ───────────────────────────────────────────
df = load_data_with_duckdb()
logger.info("已加载 %d 条记录，数据源: %s", len(df), SYSTEM_STATUS["data_source"])

# ── FastAPI 应用 ───────────────────────────────────────
app = FastAPI(
    title="大数据分析看板 API",
    description="M4 里程碑 — 系统联调交付",
    version="1.0.0",
)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "服务运行正常"}


@app.get("/api/system-status")
def get_system_status():
    """
    返回系统运行状态，包括 LLM 降级信息。
    前端根据此接口决定是否显示降级横幅。
    """
    return {
        "data_source": SYSTEM_STATUS["data_source"],
        "data_rows": SYSTEM_STATUS["data_rows"],
        "llm_active": SYSTEM_STATUS["llm_active"],
        "llm_reason": SYSTEM_STATUS["llm_reason"],
        "warnings": SYSTEM_STATUS["warnings"],
    }


@app.get("/api/category-distribution")
def get_category_distribution(sentiment: str = None):
    """
    返回各品类的样本数量。
    可选参数 sentiment：按情感筛选后再统计品类分布，实现"情感→品类"联动。
    """
    try:
        filtered = df
        if sentiment:
            filtered = filtered[filtered["sentiment"] == sentiment]
        if "cat" not in filtered.columns or filtered.empty:
            return {"categories": [], "counts": []}
        stats = filtered["cat"].value_counts()
        return {"categories": stats.index.tolist(), "counts": stats.values.tolist()}
    except Exception:
        logger.error("category-distribution 查询失败: %s", traceback.format_exc())
        return {"categories": [], "counts": [], "error": "查询失败，请检查数据源"}


@app.get("/api/sentiment-overview")
def get_sentiment_overview(cat: str = None):
    """
    返回各品类的情感分布（堆叠柱状图数据）。
    可选参数 cat：按品类筛选后仅展示该品类的情感比例。
    """
    try:
        filtered = df if cat is None else df[df["cat"] == cat]
        if filtered.empty:
            return {"data": []}
        pivot = filtered.groupby(["cat", "sentiment"]).size().unstack(fill_value=0)
        result = []
        for cat_name in pivot.index:
            entry = {"category": cat_name}
            for col in pivot.columns:
                entry[col] = int(pivot.loc[cat_name, col])
            result.append(entry)
        return {"data": result}
    except Exception:
        logger.error("sentiment-overview 查询失败: %s", traceback.format_exc())
        return {"data": [], "error": "查询失败"}


@app.get("/api/reviews")
def get_reviews(
    cat: str = None,
    sentiment: str = None,
    query: str = None,
    limit: int = 20,
):
    """
    多条件筛选评论列表。
    - cat: 品类过滤
    - sentiment: 情感过滤
    - query: 正则关键词搜索（语法错误时自动降级为普通包含匹配）
    - limit: 返回条数上限
    """
    try:
        filtered = df
        if cat:
            filtered = filtered[filtered["cat"] == cat]
        if sentiment:
            filtered = filtered[filtered["sentiment"] == sentiment]
        if query:
            try:
                filtered = filtered[
                    filtered["review"].str.contains(
                        query, case=False, na=False, regex=True
                    )
                ]
            except re.error:
                filtered = filtered[
                    filtered["review"].str.contains(
                        query, case=False, na=False, regex=False
                    )
                ]
        records = filtered.head(limit).to_dict(orient="records")
        return {"total": len(filtered), "data": records}
    except Exception:
        logger.error("reviews 查询失败: %s", traceback.format_exc())
        return {"total": 0, "data": [], "error": "查询失败"}


@app.get("/api/sub-category-stats")
def get_sub_category_stats(cat: str = Query(...)):
    """
    维度下钻接口：返回指定品类下各子维度（category 字段）的样本数量。
    """
    try:
        sub = df[df["cat"] == cat]
        if "category" not in sub.columns or sub.empty:
            return {"categories": [], "counts": []}
        stats = sub["category"].value_counts()
        return {"categories": stats.index.tolist(), "counts": stats.values.tolist()}
    except Exception:
        logger.error("sub-category-stats 查询失败: %s", traceback.format_exc())
        return {"categories": [], "counts": []}


# ── CORS 中间件 ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 静态文件服务（必须在路由之后挂载） ─────────────────
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
