FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import streamlit, numpy, shap, onnxruntime"

COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser database/ ./database/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser frontend/ ./frontend/
COPY --chown=appuser:appuser download_models.py start_space.py ./
RUN chown appuser:appuser /app

USER appuser
ENV HOME=/home/appuser PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    API_URL=http://127.0.0.1:8000
EXPOSE 7860
CMD ["python", "start_space.py"]
