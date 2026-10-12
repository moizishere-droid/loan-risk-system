"""Local integration check using existing models and a disposable SQLite DB.
Run with: python checks/smoke_single_space.py
Requires the model files already downloaded. Does not use the configured Neon DB.
"""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import requests
from download_models import BASE_DIR, FILES
from frontend.backend_client import wait_for_api


def main():
    if not all((BASE_DIR / p).is_file() for _, p in FILES):
        raise RuntimeError('Download model files first: python download_models.py')
    for port in (7860, 8000):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Port {port} is in use. Stop the existing app before this check.')
    with tempfile.TemporaryDirectory(prefix='loan-risk-check-', dir=ROOT) as tmp:
        env = os.environ.copy()
        env.update(DATABASE_URL='sqlite:///' + (Path(tmp) / 'check.db').as_posix(),
                   HF_HUB_OFFLINE='1', PYTHONIOENCODING='utf-8')
        log_path = Path(tmp) / 'services.log'
        with log_path.open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, 'start_space.py'], cwd=ROOT,
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                assert wait_for_api('http://127.0.0.1:8000', max_wait=180), 'API did not become ready'
                print('API and model/database readiness: PASS', flush=True)
                response = requests.get('http://127.0.0.1:7860/_stcore/health', timeout=10)
                assert response.status_code == 200, response.text
                print('Streamlit server: PASS', flush=True)
                from streamlit.testing.v1 import AppTest
                ui = AppTest.from_file(str(ROOT / 'frontend/app.py')).run(timeout=30)
                assert not ui.exception, ui.exception
                assert len(ui.tabs) == 4, 'Frontend did not unlock its four tabs'
                print('Frontend renders all four tabs after readiness: PASS', flush=True)
                payload = dict(cnic='99999-0000000-0', loan_amnt=5000,
                               loan_grade='A', loan_intent='EDUCATION', loan_percent_income=0.08,
                               person_income=65000, person_age=35, person_emp_length=8.0,
                               person_home_ownership='MORTGAGE', cb_person_default_on_file='N',
                               cb_person_cred_hist_length=10.0)
                base = 'http://127.0.0.1:8000'
                for path, data in [('new', payload), ('existing', {**payload, 'loan_int_rate': 12.0})]:
                    response = requests.post(f'{base}/predict/{path}', json=data, timeout=60)
                    assert response.status_code == 200, response.text
                    result = response.json()
                    assert result['classification']['decision'] in ('APPROVED', 'REJECTED')
                    assert result['regression'] is not None if path == 'new' else result['regression'] is None
                    print(f'{path} prediction with ONNX and SHAP: PASS', flush=True)
                response = requests.get(f'{base}/applicant/{payload["cnic"]}', timeout=10)
                assert response.status_code == 200 and response.json()['total_visits'] == 2, response.text
                response = requests.delete(f'{base}/applicant/{payload["cnic"]}', timeout=10)
                assert response.status_code == 200, response.text
                assert requests.get(f'{base}/applicant/{payload["cnic"]}', timeout=10).status_code == 404
                print('Applicant persistence, lookup, and delete: PASS', flush=True)
            except BaseException:
                print(log_path.read_text(encoding='utf-8', errors='replace')[-7000:])
                raise
            finally:
                # SIGTERM permits supervisor cleanup on Linux; Windows uses Ctrl+Break
                # only with a process group, so explicitly stop descendants on Windows.
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, check=True)
                else:
                    process.terminate()
                process.wait(timeout=30)


if __name__ == '__main__':
    main()
