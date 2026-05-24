# Kintsugi: adaptive spatial tessellation
# Multi-stage build for minimal image size.
#
# Build:
#   docker build -t kintsugi .
#
# Run demo:
#   docker run --rm kintsugi
#
# Interactive:
#   docker run --rm -it kintsugi python

FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY kintsugi/ kintsugi/
COPY tests/ tests/
COPY examples/ examples/

RUN pip install --no-cache-dir build && \
    python -m build --wheel && \
    pip install --no-cache-dir dist/*.whl

# --- Runtime stage ---
FROM python:3.12-slim

LABEL maintainer="Kintsugi authors"
LABEL description="Adaptive spatial tessellation for sub-cellular resolution transcriptomics"
LABEL org.opencontainers.image.source="https://github.com/cafferychen777/kintsugi"

WORKDIR /app

# Install the wheel from the builder stage
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm /tmp/*.whl

# Verify installation
RUN python -c "import kintsugi; print(f'kintsugi {kintsugi.__version__} OK')"

# Default: run the demo (bundled inside the wheel)
CMD ["python", "-m", "kintsugi.demo"]
