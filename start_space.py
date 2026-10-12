"""Run Streamlit and its private API in one Hugging Face Docker Space."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def run_api():
    # Downloads happen in this child so the UI can display startup progress.
    from download_models import download_all
    import uvicorn

    download_all()
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000)


def stop_children(children):
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def main():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    os.chdir(ROOT)
    if sys.argv[1:] == ["--api-worker"]:
        run_api()
        return 0

    env = os.environ.copy()
    # Override an old API_URL in .env or Space settings: both services are local.
    env["API_URL"] = "http://127.0.0.1:8000"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    children = []
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        commands = [
            [sys.executable, "-u", str(ROOT / "start_space.py"), "--api-worker"],
            [sys.executable, "-m", "streamlit", "run", "frontend/app.py",
             "--server.address=0.0.0.0", "--server.port=7860",
             "--server.headless=true", "--browser.gatherUsageStats=false"],
        ]
        for command in commands:
            children.append(subprocess.Popen(command, cwd=ROOT, env=env))
        while not stopping:
            for child in children:
                code = child.poll()
                if code is not None:
                    print(f"Service exited with code {code}; stopping the Space.", flush=True)
                    return code if code != 0 else 1
            time.sleep(0.5)
        return 0
    finally:
        stop_children(children)


if __name__ == "__main__":
    sys.exit(main())
