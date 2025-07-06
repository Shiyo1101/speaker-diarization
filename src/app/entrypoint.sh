#!/bin/sh
set -e

echo '🚀 FastAPI server is running at http://localhost:8000'
echo '📝 Swagger UI is available at http://localhost:8000/docs'

exec /opt/venv/bin/uvicorn src.app.main:app --host 0.0.0.0 --port 80 --reload --reload-dir src/app
