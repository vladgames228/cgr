FROM python:3.12-slim AS classic
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 tg \
    && useradd --uid 10001 --gid tg --no-create-home tg \
    && mkdir /data \
    && chown tg:tg /data
COPY uniqueness ./uniqueness
USER tg
ENTRYPOINT ["python", "-m", "uniqueness"]
CMD ["run"]

FROM classic AS semantic
USER root
COPY requirements-semantic.txt ./
RUN pip install --no-cache-dir -r requirements-semantic.txt
USER tg
