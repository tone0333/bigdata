"""
流式数据管道 — 独立运行的生产者-消费者管道 + ML 实时推理。

功能：
  - 模拟模式 (simulate): 生成合成电商行为日志
  - 回放模式 (replay): 从 CSV 流式读取历史数据
  - 队列解耦 + 背压水位线控制
  - 微批量 ML 推理打标（RandomForest 模型）
  - 死信队列隔离异常记录

用法：
  # 模拟模式
  python pipeline/streaming_pipeline.py --mode simulate --qps 50 --duration 20

  # 回放模式
  python pipeline/streaming_pipeline.py --mode replay --qps 30 --max_rows 500
"""
import argparse
import csv
import json
import os
import queue
import random
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── 把项目根目录加入路径 ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ── 颜色输出 ───────────────────────────────────────────
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ====================================================================
# 线程安全计数器
# ====================================================================
class ThreadSafeCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self._value = 0

    def increment(self, n: int = 1):
        with self._lock:
            self._value += n

    def get(self) -> int:
        with self._lock:
            return self._value


# ====================================================================
# 死信日志
# ====================================================================
class DeadLetterWriter:
    """线程安全地将残缺/异常记录追加写入 dead_letter.log"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, raw_event: Any, error: str, stage: str = "consumer"):
        record = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "error": error,
            "raw_event": str(raw_event)[:2000],
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ====================================================================
# 背压管理器
# ====================================================================
class BackpressureManager:
    """水位线背压状态机（线程安全）"""

    def __init__(self, high: float = 0.85, low: float = 0.30):
        self._lock = threading.Lock()
        self._active = False
        self.high = high
        self.low = low
        self._last_alert = 0.0
        self._cooldown = 2.0

    def update(self, depth: int, capacity: int) -> bool:
        if capacity <= 0:
            return False
        pct = depth / capacity
        with self._lock:
            if pct >= self.high and not self._active:
                self._active = True
            elif pct <= self.low and self._active:
                self._active = False
        return self._active

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def should_alert(self) -> bool:
        now = time.time()
        with self._lock:
            if now - self._last_alert >= self._cooldown:
                self._last_alert = now
                return True
            return False


# ====================================================================
# 数据生成器（模拟模式）
# ====================================================================
class DataGenerator:
    """电商行为日志生成器，Zipf 分布 + purchase 前置 view 逻辑"""

    def __init__(
        self,
        user_pool_size: int = 200,
        item_pool_size: int = 100,
        user_zipf_alpha: float = 1.2,
        item_zipf_alpha: float = 1.5,
    ):
        self.user_pool = [f"user_{i}" for i in range(1, user_pool_size + 1)]
        self.item_pool = [f"item_{i}" for i in range(1, item_pool_size + 1)]
        self.user_zipf_alpha = user_zipf_alpha
        self.item_zipf_alpha = item_zipf_alpha
        self.behavior_weights = {"view": 80, "cart": 15, "purchase": 5}
        self.user_history: Dict[str, list] = {}

    def _zipf_choice(self, pool: list, alpha: float) -> str:
        weights = [1.0 / (i ** alpha) for i in range(1, len(pool) + 1)]
        return random.choices(pool, weights=weights, k=1)[0]

    def _choose_behavior(self, user_id: str) -> str:
        recent = self.user_history.get(user_id, [])
        has_view = "view" in recent[-5:] if recent else False
        if has_view:
            choices = list(self.behavior_weights.keys())
            weights = list(self.behavior_weights.values())
        else:
            choices = list(self.behavior_weights.keys())
            weights = [85, 14, 1]
        return random.choices(choices, weights=weights, k=1)[0]

    def generate(self) -> Dict[str, Any]:
        user_id = self._zipf_choice(self.user_pool, self.user_zipf_alpha)
        item_id = self._zipf_choice(self.item_pool, self.item_zipf_alpha)
        behavior_type = self._choose_behavior(user_id)

        if user_id not in self.user_history:
            self.user_history[user_id] = []
        self.user_history[user_id].append(behavior_type)
        if len(self.user_history[user_id]) > 10:
            self.user_history[user_id] = self.user_history[user_id][-10:]

        time_window = int(time.time() // 1800)
        session_id = f"sess_{user_id}_{time_window}_{uuid.uuid4().hex[:6]}"
        timestamp = int(time.time())

        return {
            "user_id": user_id,
            "item_id": item_id,
            "session_id": session_id,
            "behavior_type": behavior_type,
            "timestamp": str(timestamp),
        }


# ====================================================================
# 模型加载器
# ====================================================================
class ModelLoader:
    """单例模型加载器"""
    _instance = None
    _model = None

    @classmethod
    def get_model(cls, model_path: Path | None):
        if cls._instance is None and model_path and model_path.exists():
            cls._instance = ModelLoader()
            cls._model = cls._instance._load(model_path)
        return cls._model

    def _load(self, model_path: Path):
        import joblib
        print(f"[ModelLoader] 加载模型: {model_path}  "
              f"({os.path.getsize(model_path)/1024/1024:.2f} MB)")
        t0 = time.perf_counter()
        model = joblib.load(str(model_path))
        print(f"[ModelLoader] 加载完成，耗时 {time.perf_counter() - t0:.3f}s")
        return model


# ====================================================================
# 特征提取
# ====================================================================
def extract_features_batch(events: List[Dict[str, Any]]) -> pd.DataFrame:
    """批量特征提取，单个事件失败用默认值填充而不中断整批"""
    rows = []
    for ev in events:
        try:
            rows.append({
                "user_id": str(ev.get("user_id", "0")),
                "item_id": str(ev.get("item_id", "0")),
                "session_id": str(ev.get("session_id", "0")),
                "timestamp": int(float(str(ev.get("timestamp", "0")))),
            })
        except Exception:
            rows.append(
                {"user_id": "0", "item_id": "0", "session_id": "0", "timestamp": 0}
            )
    return pd.DataFrame(rows)


# ====================================================================
# 输出写入器
# ====================================================================
class OutputWriter:
    """线程安全 CSV 输出"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._initialized = False
        self._fieldnames: List[str] = []

    def write(self, event: Dict[str, Any]):
        with self._lock:
            if not self._initialized:
                self._fieldnames = [
                    "user_id", "item_id", "session_id",
                    "behavior_type", "timestamp",
                    "predicted_label", "buy_probability", "error",
                ]
                with open(self.path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=self._fieldnames)
                    w.writeheader()
                self._initialized = True

            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._fieldnames)
                slim = {k: event.get(k, "") for k in self._fieldnames}
                w.writerow(slim)


# ====================================================================
# Producer
# ====================================================================
def producer(
    q: queue.Queue,
    mode: str,
    qps: int,
    max_rows: int,
    dataset_path: Path | None,
    backpressure_mgr: BackpressureManager,
    produced_counter: ThreadSafeCounter,
    stop_event: threading.Event,
):
    """Producer 线程入口"""
    if mode == "replay" and dataset_path and dataset_path.exists():
        _producer_replay(q, dataset_path, qps, max_rows,
                         produced_counter, stop_event)
    else:
        _producer_simulate(q, qps, max_rows, backpressure_mgr,
                           produced_counter, stop_event)


def _producer_simulate(q, qps, max_rows, backpressure_mgr, produced_counter, stop_event):
    print(f"[Producer] 模拟模式 | 速率 {qps} 条/秒 | 上限 {max_rows} 条")
    generator = DataGenerator()
    interval = 1.0 / qps
    current_delay = interval

    while not stop_event.is_set() and produced_counter.get() < max_rows:
        entry = generator.generate()
        try:
            q.put(entry, block=True, timeout=0.5)
            produced_counter.increment()
            if not backpressure_mgr.is_active():
                current_delay = max(current_delay / 2, interval)
        except queue.Full:
            current_delay = min(current_delay * 2, 2.0)
            continue
        time.sleep(current_delay)

    print(f"[Producer] 停止 | 产量 {produced_counter.get()}")


def _producer_replay(q, dataset_path, qps, max_rows, produced_counter, stop_event):
    print(f"[Producer] 回放模式 | {dataset_path} | 速率 {qps} 条/秒")
    interval = 1.0 / qps
    rows_read = 0

    for enc in ["utf-8", "gbk", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            with open(dataset_path, "r", encoding=enc) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if stop_event.is_set() or rows_read >= max_rows:
                        break
                    try:
                        q.put(row, block=True, timeout=1.0)
                        produced_counter.increment()
                        rows_read += 1
                        time.sleep(interval)
                    except queue.Full:
                        continue
                print(f"[Producer] 编码 {enc} 完成 | 产量 {produced_counter.get()}")
                return
        except (UnicodeDecodeError, PermissionError, Exception):
            continue

    print("[Producer] 编码全部失败，回退到模拟模式")
    generator = DataGenerator()
    interval = 1.0 / qps
    while not stop_event.is_set() and produced_counter.get() < max_rows:
        entry = generator.generate()
        try:
            q.put(entry, block=True, timeout=0.5)
            produced_counter.increment()
        except queue.Full:
            continue
        time.sleep(interval)


# ====================================================================
# Consumer
# ====================================================================
def consumer(
    q: queue.Queue,
    model: Any,
    batch_size: int,
    batch_timeout: float,
    consumed_counter: ThreadSafeCounter,
    consumer_id: int,
    stop_event: threading.Event,
    output_writer: OutputWriter,
    dead_letter: DeadLetterWriter,
):
    """Consumer 线程：微批量推理，坏数据写入死信"""
    print(f"[Consumer-{consumer_id}] 启动 | batch={batch_size}")

    buffer: List[Dict[str, Any]] = []
    last_flush = time.time()

    while not stop_event.is_set():
        try:
            event = q.get(timeout=0.1)
        except queue.Empty:
            event = None

        if event is not None:
            buffer.append(event)

        now = time.time()
        should_flush = len(buffer) >= batch_size or (
            buffer and (now - last_flush > batch_timeout)
        )

        if should_flush:
            _flush_batch(q, buffer, model, consumed_counter, consumer_id,
                         output_writer, dead_letter)
            buffer.clear()
            last_flush = now

    if buffer:
        _flush_batch(q, buffer, model, consumed_counter, consumer_id,
                     output_writer, dead_letter)

    print(f"[Consumer-{consumer_id}] 停止 | 总消费 {consumed_counter.get()}")


def _flush_batch(
    q: queue.Queue,
    buffer: List[Dict[str, Any]],
    model: Any,
    consumed_counter: ThreadSafeCounter,
    consumer_id: int,
    output_writer: OutputWriter,
    dead_letter: DeadLetterWriter,
):
    """处理一个微批次 — 绝不让进程崩溃"""
    batch_size = len(buffer)
    try:
        features = extract_features_batch(buffer)

        if model is not None:
            predicted_labels = model.predict(features)
            predicted_probs = model.predict_proba(features)
        else:
            predicted_labels = [-1] * len(buffer)
            predicted_probs = [[0.0, 0.0]] * len(buffer)

        for i, event in enumerate(buffer):
            scored = dict(event)
            try:
                scored["predicted_label"] = int(predicted_labels[i])
                scored["buy_probability"] = float(predicted_probs[i][1])
                scored["error"] = ""
            except Exception as inner_e:
                scored["predicted_label"] = -1
                scored["buy_probability"] = -1.0
                scored["error"] = str(inner_e)
                dead_letter.write(event, str(inner_e), stage="result_mapping")

            output_writer.write(scored)
            consumed_counter.increment()
            q.task_done()

    except Exception as batch_e:
        print(f"[Consumer-{consumer_id}] 批次失败 ({batch_size}条): {batch_e}")
        for event in buffer:
            scored = dict(event)
            scored["predicted_label"] = -1
            scored["buy_probability"] = -1.0
            scored["error"] = f"batch_failure: {batch_e}"
            dead_letter.write(event, str(batch_e), stage="batch_inference")
            output_writer.write(scored)
            consumed_counter.increment()
            try:
                q.task_done()
            except Exception:
                pass


# ====================================================================
# 主框架
# ====================================================================
def run_pipeline(args: argparse.Namespace):
    """启动 Producer + Consumer + 背压监控"""
    print()
    print(CYAN + BOLD + "=" * 58 + RESET)
    print(CYAN + BOLD + "  流式数据处理管道" + RESET)
    print(CYAN + BOLD + "=" * 58 + RESET)
    print(f"  模式: {args.mode}  QPS: {args.qps}  "
          f"Consumer: {args.consumers}  批次: {args.batch_size}")
    print(f"  背压: {'启用' if args.backpressure else '禁用'}  "
          f"时长: {args.duration}s  最大行数: {args.max_rows}")
    print(CYAN + "=" * 58 + RESET)

    # 1. 加载模型
    model = None
    model_path = args.model
    if not model_path:
        model_path = config.resolve_model_path()
    if model_path and model_path.exists():
        model = ModelLoader.get_model(model_path)
    else:
        print(YELLOW + "[WARN] 未加载模型，Consumer 将跳过推理仅做透传" + RESET)

    # 2. 创建组件
    q = queue.Queue(maxsize=args.queue_limit)
    produced_counter = ThreadSafeCounter()
    consumed_counter = ThreadSafeCounter()
    stop_event = threading.Event()

    output_dir = config.DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_writer = OutputWriter(output_dir / "scored_events.csv")
    dead_letter = DeadLetterWriter(output_dir / "dead_letter.log")
    backpressure_mgr = BackpressureManager(
        config.BACKPRESSURE_HIGH, config.BACKPRESSURE_LOW
    )

    start_time = time.time()

    # 3. 启动 Producer
    p = threading.Thread(
        target=producer,
        args=(
            q, args.mode, args.qps, args.max_rows,
            Path(args.dataset) if args.dataset else None,
            backpressure_mgr, produced_counter, stop_event,
        ),
        name="Producer", daemon=True,
    )
    p.start()

    # 4. 启动 Consumers
    consumers_list = []
    for i in range(args.consumers):
        c = threading.Thread(
            target=consumer,
            args=(
                q, model, args.batch_size, args.batch_timeout,
                consumed_counter, i + 1, stop_event,
                output_writer, dead_letter,
            ),
            name=f"Consumer-{i+1}", daemon=True,
        )
        c.start()
        consumers_list.append(c)

    # 5. 等待 + 监控
    try:
        print(f"\n  流水线运行中 ...\n" + "-" * 58)
        while time.time() - start_time < args.duration:
            time.sleep(0.5)
            depth = q.qsize()
            pct = (f"{depth / args.queue_limit * 100:.1f}%"
                   if args.queue_limit > 0 else "∞")

            if args.backpressure:
                activated = backpressure_mgr.update(depth, args.queue_limit)
                if backpressure_mgr.should_alert():
                    status = "[BACKPRESSURE ON]" if activated else "[NORMAL]"
                    elapsed = time.time() - start_time
                    print(
                        f"  [{elapsed:.0f}s] 队列: {depth}/{args.queue_limit}"
                        f" ({pct}) {status} | 产:{produced_counter.get()}"
                        f" 消:{consumed_counter.get()}"
                    )

            if (produced_counter.get() >= args.max_rows
                    and consumed_counter.get() >= produced_counter.get()):
                print(f"  全部 {args.max_rows} 条数据处理完毕，提前结束")
                break

    except KeyboardInterrupt:
        print("\n  收到中断信号")

    # 6. 停止
    print("\n  正在停止 ...")
    stop_event.set()
    p.join(timeout=3)
    for c in consumers_list:
        c.join(timeout=3)

    # 7. 统计
    elapsed = time.time() - start_time
    print("\n" + "=" * 58)
    print("  统计摘要")
    print("=" * 58)
    print(f"  运行时长: {elapsed:.1f}s")
    print(f"  生产总量: {produced_counter.get()}")
    print(f"  消费总量: {consumed_counter.get()}")
    print(f"  最终队列深度: {q.qsize()}")
    print(f"  输出: {output_dir / 'scored_events.csv'}")
    print(f"  死信: {output_dir / 'dead_letter.log'}")

    dl_path = output_dir / "dead_letter.log"
    if dl_path.exists():
        try:
            with open(dl_path, "r", encoding="utf-8") as f:
                dl_count = sum(1 for _ in f)
            print(f"  死信记录数: {dl_count}")
        except Exception:
            pass
    print("=" * 58)


def main():
    parser = argparse.ArgumentParser(
        description="流式数据处理管道 — 模拟/回放 -> 队列 -> ML推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python pipeline/streaming_pipeline.py --mode simulate --qps 50 --duration 20
  python pipeline/streaming_pipeline.py --mode replay --qps 30 --max_rows 500
        """,
    )
    parser.add_argument("--mode", choices=["simulate", "replay"], default="simulate")
    parser.add_argument("--qps", type=int, default=config.PIPELINE_QPS)
    parser.add_argument("--queue_limit", type=int, default=config.PIPELINE_QUEUE_LIMIT)
    parser.add_argument("--consumers", type=int, default=config.PIPELINE_CONSUMERS)
    parser.add_argument("--batch_size", type=int, default=config.PIPELINE_BATCH_SIZE)
    parser.add_argument(
        "--batch_timeout", type=float, default=config.PIPELINE_BATCH_TIMEOUT
    )
    parser.add_argument("--duration", type=int, default=config.PIPELINE_DURATION)
    parser.add_argument("--max_rows", type=int, default=500)
    parser.add_argument("--dataset", type=str, default=None,
                        help="回放模式的 CSV 数据源路径")
    parser.add_argument("--model", type=str, default=None,
                        help="模型文件路径（自动查找）")
    parser.add_argument("--backpressure", type=lambda x: x.lower() in ("true", "1", "yes"),
                        default=True)
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
