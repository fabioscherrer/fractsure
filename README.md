# Fractsure MLOps Monorepo

Clean scaffold for an end-to-end fracture detection pipeline with clear separation between:

- Model development and training (`training/`)
- Inference serving (`api/`)
- User-facing UI (`frontend/`)

## Stack

- Package management: `uv`
- ML: YOLO (Ultralytics), MLflow, ONNX Runtime
- Serving: FastAPI
- Frontend: Streamlit
- Data versioning: DVC
- Containerization: Docker + Docker Compose
- CI: GitHub Actions

## Repository Layout

```text
.
├── .github/workflows/ci.yml
├── .dvc/
├── training/
│   ├── config.yaml
│   ├── export.py
│   └── train.py
├── api/
│   ├── Dockerfile
│   ├── main.py
│   └── model/
├── frontend/
│   ├── app.py
│   └── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

## Local Development with uv

1. Install dependencies:

```bash
uv sync
```

2. Run the API:

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

3. In another terminal, run the frontend:

```bash
API_URL=http://localhost:8000 uv run streamlit run frontend/app.py
```

## Docker Compose

Run both services:

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Frontend: `http://localhost:8501`

The frontend uses `API_URL=http://api:8000` inside the Docker network.

Quick API smoke test from the host after `docker compose up --build`:

```bash
curl http://localhost:8000/health
```

```bash
# bash/zsh
TEST_IMAGE="$(find data/raw/hbfmid/test/images -type f | head -n 1)"
curl -X POST http://localhost:8000/predict -F "file=@${TEST_IMAGE}"

# fish
set TEST_IMAGE (find data/raw/hbfmid/test/images -type f | head -n 1)
curl -X POST http://localhost:8000/predict -F "file=@$TEST_IMAGE"
```

Expected behavior:

- `health` returns `{"status":"ok"}`.
- `predict` returns JSON with `model_loaded` and `boxes`.
- The API uses `CONF_THRESHOLD=0.01` by default for this exported model; tune with `CONF_THRESHOLD` env if detections are too sparse or too noisy.
- If no ONNX model is present in `api/model/`, the API still returns a placeholder box (`model_loaded: false`).
- If you export an ONNX file after containers were built, rebuild the API image so the container can see it (`docker compose up --build api`).

## Training Notes

- Training configuration lives in `training/config.yaml`.
- The dataset path inside `training/config.yaml` should be updated to your local/project dataset setup.
- Run training:

```bash
uv run python training/train.py
```

- Export the best run to ONNX:

```bash
uv run python training/export.py
```

Exported models are written to `api/model/`.

## Data and DVC

- This repository includes only the DVC scaffold under `.dvc/`.
- Dataset files and tracking metadata should be set up fresh for this repository.
- Large datasets, model weights, and local experiment artifacts should not be committed to Git.

## Project Goal

This repository is intended to be the clean shared version of the fracture detection project for collaboration with teammates and later submission.