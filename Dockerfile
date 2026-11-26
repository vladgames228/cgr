# syntax=docker/dockerfile:1
FROM python:3.12-slim AS model
WORKDIR /model
ADD --checksum=sha256:98a01d88b7de996cdea58c32ca71208c09968d143798814b2ea09d3439dc334f https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/e8f8c211226b894fcb81acc59f3b34ba3efd5f42/onnx/model_quint8_avx2.onnx model.onnx
ADD --checksum=sha256:2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8 https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/e8f8c211226b894fcb81acc59f3b34ba3efd5f42/tokenizer.json tokenizer.json

FROM python:3.12-slim AS semantic
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MODEL_DIR=/model
WORKDIR /app
COPY requirements.txt requirements-semantic.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-semantic.txt \
    && groupadd --gid 10001 tg \
    && useradd --uid 10001 --gid tg --no-create-home tg \
    && mkdir /data \
    && chown tg:tg /data
COPY --from=model --chmod=0444 /model/model.onnx /model/tokenizer.json /model/
COPY uniqueness ./uniqueness
USER tg
ENTRYPOINT ["python", "-m", "uniqueness"]
CMD ["run"]
