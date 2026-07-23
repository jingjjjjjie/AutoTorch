ARG BASE_IMAGE=idrecapture_automl:v1.0.0
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/idrecapture_server:/idrecapture_server/idfraud

WORKDIR /app

COPY advanced_visualization/tensorflow_service/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY advanced_visualization/tensorflow_service /app/advanced_visualization/tensorflow_service

EXPOSE 8701

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8701/health')"

ENTRYPOINT []
CMD ["uvicorn", "advanced_visualization.tensorflow_service.app:app", "--host", "0.0.0.0", "--port", "8701", "--workers", "1"]
