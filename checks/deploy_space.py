"""Deploy the combined app from the root, using credentials from local .env."""
from pathlib import Path
import argparse
import os

from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = 'Abdulmoiz123/loan-risk-app'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs-only', action='store_true')
    args = parser.parse_args()
    settings = dotenv_values(ROOT / '.env')
    token = os.environ.get('HF_TOKEN') or settings.get('HF_TOKEN')
    database_url = os.environ.get('DATABASE_URL') or settings.get('DATABASE_URL')
    if not token or not database_url:
        raise SystemExit('Set HF_TOKEN and DATABASE_URL in .env before deploying.')
    if not database_url.startswith(('postgresql://', 'postgresql+psycopg2://')):
        raise SystemExit('Deployment needs a PostgreSQL DATABASE_URL.')
    api = HfApi(token=token)
    identity = api.whoami()
    print('Deploying as:', identity['name'], flush=True)
    info = api.space_info(SPACE_ID)
    if info.sdk != 'docker':
        raise SystemExit('The target Space must use the Docker SDK.')
    if args.docs_only:
        result = api.upload_file(repo_id=SPACE_ID, repo_type='space',
                                 path_or_fileobj=ROOT / 'SINGLE_SPACE_GUIDE.md',
                                 path_in_repo='SINGLE_SPACE_GUIDE.md',
                                 commit_message='Document single-Space setup and verification')
        print('Documentation uploaded:', result.oid, flush=True)
        return
    api.add_space_secret(SPACE_ID, key='DATABASE_URL', value=database_url)
    # Keep model-download credentials server-side; never copy .env into the image.
    api.add_space_secret(SPACE_ID, key='HF_TOKEN', value=token)
    result = api.upload_folder(
        repo_id=SPACE_ID, repo_type='space', folder_path=ROOT,
        allow_patterns=['Dockerfile', 'README.md', 'SINGLE_SPACE_GUIDE.md', '.dockerignore',
                        'start_space.py', 'download_models.py',
                        'requirements.txt', 'api/*.py', 'database/*.py',
                        'src/*.py', 'frontend/*.py', 'frontend/ui/*.css'],
        # Delete only superseded deployment files, never broad repository contents.
        delete_patterns=['requirements.api.txt', 'requirements.frontend.txt',
                         'requirements.space.txt', 'requirements_frozen.txt',
                         'Dockerfile.api', 'Dockerfile.frontend'],
        commit_message='Add professional Streamlit frontend and private account workspaces',
    )
    print('Uploaded commit:', result.oid, flush=True)
    print('Space: https://huggingface.co/spaces/' + SPACE_ID, flush=True)
    print('Application: https://abdulmoiz123-loan-risk-app.hf.space', flush=True)


if __name__ == '__main__':
    main()
