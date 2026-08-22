FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY config.yaml ./config.yaml

RUN pip install --no-cache-dir --editable ".[dev]"

ENTRYPOINT ["python", "-m", "palace.cli.app"]
