"""
一键启动脚本 — 实验十四系统联调。

功能：
  1. 环境自检：数据文件、前端文件、端口占用
  2. 异步子进程启动 FastAPI (uvicorn)
  3. HTTP 轮询等待服务就绪后自动打开浏览器
  4. Ctrl+C 优雅关闭，不遗留孤儿进程

用法：
  python run_app.py
  python run_app.py --port 8080 --no-browser
"""
import argparse
import http.client
import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# ── 把项目根目录加入路径 ─────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402

# ── 终端颜色 ───────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Windows GBK 终端兼容 — 使用 ASCII 安全符号
OK = "[OK]"
WARN = "[WARN]"
FAIL = "[FAIL]"


def print_header():
    print()
    print(CYAN + BOLD + "=" * 58 + RESET)
    print(CYAN + BOLD + "   大数据分析看板 · 系统联调启动脚本" + RESET)
    print(CYAN + BOLD + "=" * 58 + RESET)
    print()


def env_check() -> bool:
    """环境自检：检查必要文件和端口，返回是否通过。"""
    print(BOLD + "[1/4] 环境自检 ..." + RESET)
    all_ok = True

    # 检查数据文件
    if config.check_data_file(config.FEATURES_CSV):
        print(GREEN + f"  {OK} LLM 特征数据: {config.FEATURES_CSV.name}" + RESET)
    elif config.RAW_CSV.exists():
        print(YELLOW + f"  {WARN} LLM 特征数据缺失，将回退到原始 CSV" + RESET)
    else:
        print(RED + f"  {FAIL} 数据文件缺失！请将 batch_1000_features.csv 放入 data/ 目录" + RESET)
        all_ok = False

    # 检查前端文件
    frontend_html = config.FRONTEND_DIR / "index.html"
    if frontend_html.exists():
        print(GREEN + f"  {OK} 前端页面: {frontend_html.name}" + RESET)
    else:
        print(RED + f"  {FAIL} 前端文件缺失！" + RESET)
        all_ok = False

    # 检查端口占用
    if _is_port_in_use(config.HOST, config.PORT):
        print(
            RED + f"  {FAIL} 端口 {config.PORT} 已被占用！"
            f"请先关闭占用进程或使用 --port 指定其他端口" + RESET
        )
        all_ok = False
    else:
        print(GREEN + f"  {OK} 端口 {config.PORT} 空闲" + RESET)

    # 检查 LLM API Key
    llm_status = config.check_llm_api_key()
    if llm_status["active"]:
        print(GREEN + f"  {OK} 已检测到 SILICONFLOW_API_KEY" + RESET)
    else:
        print(
            YELLOW + f"  {WARN} 未检测到 SILICONFLOW_API_KEY，"
            "LLM 功能将处于降级模式" + RESET
        )

    if not all_ok:
        print()
        print(RED + "环境自检未通过，请修复上述问题后重试。" + RESET)
    return all_ok


def _is_port_in_use(host: str, port: int) -> bool:
    """检测端口是否被占用。"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False


def start_server(host: str, port: int) -> subprocess.Popen:
    """启动 uvicorn 子进程，返回 Popen 对象。"""
    print()
    print(BOLD + "[2/4] 启动 FastAPI 服务 ..." + RESET)
    print(f"  uvicorn server:app --host {host} --port {port}")

    # Windows 下 CREATE_NEW_PROCESS_GROUP 用于 Ctrl+C 时的进程组管理
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "server:app",
            "--host", host,
            "--port", str(port),
        ],
        cwd=str(PROJECT_DIR),
        creationflags=creationflags,
        # 不捕获 stdout/stderr，让 uvicorn 直接输出到控制台
    )
    return proc


def wait_for_ready(url: str, timeout: int = 30, interval: float = 0.5) -> bool:
    """
    HTTP GET 轮询等待服务就绪。
    返回 True 表示就绪，False 表示超时。
    """
    print()
    print(BOLD + "[3/4] 等待服务就绪 ..." + RESET, end="", flush=True)

    start = time.time()
    while time.time() - start < timeout:
        try:
            conn = http.client.HTTPConnection(config.HOST, config.PORT, timeout=2)
            conn.request("GET", "/api/health")
            resp = conn.getresponse()
            if resp.status == 200:
                elapsed = time.time() - start
                print(f"\r" + GREEN + f"[3/4] 服务就绪 {OK} (耗时 {elapsed:.1f}s)" + RESET)
                conn.close()
                return True
            conn.close()
        except (ConnectionRefusedError, OSError, http.client.HTTPException):
            pass

        print(".", end="", flush=True)
        time.sleep(interval)

    print()
    print(RED + f"  服务启动超时 ({timeout}s)，请检查 uvicorn 输出" + RESET)
    return False


def open_browser(url: str):
    """自动打开浏览器。"""
    print()
    print(BOLD + "[4/4] 打开浏览器 ..." + RESET)
    print(f"  {url}")
    webbrowser.open(url)
    print(GREEN + f"  {OK} 浏览器已打开" + RESET)


def cleanup(proc: subprocess.Popen | None):
    """优雅关闭子进程。"""
    if proc is None:
        return
    print()
    print(YELLOW + "\n正在关闭服务 ..." + RESET)
    if sys.platform == "win32":
        # Windows: 发送 Ctrl+Break 信号到进程组
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(RED + "进程未响应，强制终止 ..." + RESET)
            proc.kill()
            proc.wait()
    else:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    print(GREEN + f"{OK} 服务已关闭" + RESET)


def main():
    parser = argparse.ArgumentParser(description="大数据分析看板 — 一键启动脚本")
    parser.add_argument("--port", type=int, default=config.PORT, help="服务端口")
    parser.add_argument("--host", type=str, default=config.HOST, help="绑定地址")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    # 覆盖 config 中的端口配置
    config.PORT = args.port
    config.HOST = args.host
    url = f"http://{args.host}:{args.port}"

    print_header()

    # 1. 环境自检
    if not env_check():
        sys.exit(1)

    proc = None
    try:
        # 2. 启动服务
        proc = start_server(args.host, args.port)

        # 3. 等待就绪
        if not wait_for_ready(url):
            cleanup(proc)
            sys.exit(1)

        # 4. 打开浏览器
        if not args.no_browser:
            open_browser(url)

        print()
        print(GREEN + BOLD + "=" * 58 + RESET)
        print(GREEN + BOLD + "  系统运行中！按 Ctrl+C 停止" + RESET)
        print(GREEN + BOLD + "=" * 58 + RESET)

        # 等待子进程结束（或用户 Ctrl+C）
        proc.wait()

    except KeyboardInterrupt:
        cleanup(proc)
        print("\n下次再见！")
        sys.exit(0)
    except Exception as e:
        print(RED + f"\n启动异常: {e}" + RESET)
        cleanup(proc)
        sys.exit(1)


if __name__ == "__main__":
    main()
