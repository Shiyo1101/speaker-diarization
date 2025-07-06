from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.app.models.diarization import DiarizationResponse
from src.app.services.transcription import DiarizationService

router = APIRouter()
diarization_service = DiarizationService()


@router.post("/diarize", tags=["Diarization"])
async def diarize_audio(file: Annotated[UploadFile, File(...)]) -> DiarizationResponse:
    """音声ファイルをアップロードして話者分類と文字起こしを実行する.

    対応フォーマット: mp4, mp3, wav
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing.")

    allowed_content_types = ["audio/wav", "audio/mpeg", "video/mp4", "audio/x-m4a"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
            "Supported types are mp4, mp3, wav.",
        )

    try:
        return await diarization_service.process_audio(file, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}") from e
