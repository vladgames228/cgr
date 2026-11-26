# syntax=docker/dockerfile:1
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt requirements-semantic.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-semantic.txt \
    && groupadd --gid 10001 tg \
    && useradd --uid 10001 --gid tg --no-create-home tg \
    && mkdir /data \
    && chown tg:tg /data
COPY uniqueness ./uniqueness
USER tg
ENTRYPOINT ["python", "-m", "uniqueness"]
CMD ["run"]
