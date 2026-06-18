from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


HOST = "127.0.0.1"
PORT = 8502
APP_FILE = "app.py"


def app_url() -> str:
    return f"http://{HOST}:{PORT}"


def health_url() -> str:
    return f"{app_url()}/_stcore/health"


def is_running() -> bool:
    """检查 Streamlit 服务是否已在运行。"""
    try:
        with urllib.request.urlopen(health_url(), timeout=1.5) as response:
            return response.read().decode("utf-8", errors="ignore").strip() == "ok"
    except (urllib.error.URLError, TimeoutError):
        return False


def wait_until_ready(seconds: int = 30) -> bool:
    """等待 Streamlit 服务启动就绪。"""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if is_running():
            return True
        time.sleep(0.5)
    return False


def start_streamlit(project_dir: Path) -> subprocess.Popen[str]:
    """启动 Streamlit 子进程。"""
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        APP_FILE,
        "--server.address",
        HOST,
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.Popen(command, cwd=project_dir)


def main() -> int:
    """启动 Streamlit 演示看板并打开浏览器。"""
    project_dir = Path(__file__).resolve().parent
    app_path = project_dir / APP_FILE
    if not app_path.exists():
        print(f"找不到 {app_path}")
        return 1

    if is_running():
        print(f"Web 应用已在运行: {app_url()}")
        webbrowser.open(app_url())
        return 0

    print("正在启动 Streamlit Web 应用...")
    process = start_streamlit(project_dir)
    if not wait_until_ready():
        print("启动超时，请检查 Streamlit 是否已安装：")
        print(f"{sys.executable} -m pip install -r requirements.txt")
        process.terminate()
        return 1

    print(f"正在打开 {app_url()}")
    webbrowser.open(app_url())
    print("请保持此窗口运行。按 Ctrl+C 停止。")

    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print("\nWeb 应用已停止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
