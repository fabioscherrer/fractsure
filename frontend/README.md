# Fractsure Frontend

React/Vite single-page frontend for the fracture detection API.

## Structure

```text
frontend/
├── src/
│   ├── api/              # API calls
│   ├── components/       # Small UI building blocks
│   ├── styles/           # Global and page styles
│   ├── App.jsx           # Single-page composition
│   ├── config.js         # Frontend configuration
│   └── main.jsx          # React entrypoint
├── index.html
├── vite.config.js
├── nginx.conf            # Docker proxy for /api
└── Dockerfile
```

## Local Development

Start the API from the repository root:

```powershell
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:8501`.

The app calls `/api/*`; Vite proxies those requests to `http://localhost:8000`.
