from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.app.models.diarization import DiarizationResponse
from src.app.services.transcription import DiarizationService

router = APIRouter()


# サービスのインスタンスを一度だけ生成し、キャッシュする関数を作成
@lru_cache(maxsize=1)
def get_diarization_service() -> DiarizationService:
    """話者分類サービス(DiarizationService)のインスタンスを取得する.

    lru_cacheデコレータにより、初回呼び出し時にのみインスタンスが生成され、
    以降はそのキャッシュが返されるシングルトンパターンとして機能します。

    Returns:
        DiarizationService: サービスのシングルトンインスタンス

    """
    return DiarizationService()


@router.post("/diarize", tags=["Diarization"])
async def diarize_audio(
    file: Annotated[UploadFile, File(...)],
    # Dependsを使って、エンドポイントが呼び出されたときにサービスを取得
    service: Annotated[DiarizationService, Depends(get_diarization_service)],
) -> DiarizationResponse:
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
        return await service.process_audio(file, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}") from e
