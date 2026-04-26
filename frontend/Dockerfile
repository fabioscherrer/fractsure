FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md /app/
RUN uv sync --no-install-project

COPY frontend /app/frontend

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
