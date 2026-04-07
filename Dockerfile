FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY artifacts ./artifacts
COPY artifacts_final ./artifacts_final
COPY model_service ./model_service
COPY scripts ./scripts

EXPOSE 5000
EXPOSE 5001

CMD ["python", "app.py"]
