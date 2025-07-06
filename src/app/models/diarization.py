# src/models/diarization.py


from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    """個々の発言セグメントのデータモデル."""

    speaker: str = Field(..., description="話者のラベル(例: 'SPEAKER_00')")
    text: str = Field(..., description="話者が発言したテキスト")
    start: float = Field(..., description="発言の開始時間(秒)")
    end: float = Field(..., description="発言の終了時間(秒)")


class DiarizationResponse(BaseModel):
    """話者分類APIのレスポンスのデータモデル."""

    transcription: list[TranscriptionSegment]
