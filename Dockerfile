# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS builder
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir build && python -m build --wheel

FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 reflex
WORKDIR /app
COPY --from=builder /src/dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/*.whl && python -m pip install --no-cache-dir 'agent-reflex[server]' && rm -f /tmp/*.whl
COPY examples ./examples
USER reflex
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)" || exit 1
CMD ["agent-reflex", "serve", "--factory", "examples.server_app:build_engine", "--host", "0.0.0.0", "--port", "8080"]
