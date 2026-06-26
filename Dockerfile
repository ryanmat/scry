# Description: Container image for serving the Scry prediction API.
# Description: Installs the core package and runs the FastAPI app via uvicorn.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/models/xdec_model.pt

WORKDIR /app

# Install the package (core deps only; add extras as needed).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY config ./config
RUN pip install --upgrade pip && pip install .

# Provide model weights at runtime, e.g. -v $(pwd)/models:/models
EXPOSE 8000
CMD ["uvicorn", "scry.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
