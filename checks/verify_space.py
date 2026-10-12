"""Check the live combined Space and optionally pause the superseded API."""
import argparse
from pathlib import Path
import os
import time

from dotenv import dotenv_values
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError
import requests

ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = 'Abdulmoiz123/loan-risk-app'
APP_URL = 'https://abdulmoiz123-loan-risk-app.hf.space'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pause-old', action='store_true')
    args = parser.parse_args()
    token = os.environ.get('HF_TOKEN') or dotenv_values(ROOT / '.env').get('HF_TOKEN')
    api = HfApi(token=token)
    runtime = api.get_space_runtime(SPACE_ID)
    print('Combined Space stage:', runtime.stage, flush=True)
    if runtime.stage != 'RUNNING':
        return 2
    response = requests.get(APP_URL + '/_stcore/health', timeout=20)
    response.raise_for_status()
    print('Public Streamlit health:', response.text.strip(), flush=True)
    ready = False
    started = time.monotonic()
    try:
        with requests.get(f'https://huggingface.co/api/spaces/{SPACE_ID}/logs/run',
                          headers={'Authorization': f'Bearer {token}'},
                          stream=True, timeout=(10, 20)) as logs:
            logs.raise_for_status()
            for line in logs.iter_lines(chunk_size=1, decode_unicode=True):
                if isinstance(line, bytes):
                    line = line.decode('utf-8', errors='replace')
                if 'All models loaded successfully' in line:
                    ready = True
                    break
                if time.monotonic() - started > 30:
                    break
    except requests.RequestException:
        pass
    if not ready:
        print('API model/database startup confirmation is not available yet.', flush=True)
        return 2
    print('Live API startup: models loaded after successful database initialization.', flush=True)
    if args.pause_old:
        try:
            previous = api.get_space_runtime('Abdulmoiz123/loan-risk-api')
        except RepositoryNotFoundError:
            print('Old separate API Space: already removed', flush=True)
        else:
            if previous.stage != 'PAUSED':
                api.pause_space('Abdulmoiz123/loan-risk-api')
            print('Old separate API Space: PAUSED', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
