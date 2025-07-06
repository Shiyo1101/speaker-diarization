from fastapi import FastAPI

from src.app.routers import diarization

app = FastAPI(
    title="Speaker Diarization API",
    description="pyannote-audioとAWS Transcribeを使用した話者分類と文字起こしAPI",
    version="1.0.0",
)

app.include_router(diarization.router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def read_root() -> dict:
    """ルートエンドポイントのハンドラー."""
    return {"message": "Welcome to the Speaker Diarization API!"}
