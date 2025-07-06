import asyncio
import logging
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

import aiofiles
import boto3
import httpx
import pandas as pd
from botocore.exceptions import ClientError
from fastapi import UploadFile
from pyannote.audio import Pipeline
from pydub import AudioSegment

from src.app.core.settings import settings
from src.app.models.diarization import DiarizationResponse, TranscriptionSegment

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 5
JOB_TIMEOUT_S = 600
SPEECH_MERGE_THRESHOLD_S = 0.5  # 同じ話者の発言を結合する最大の間隔(秒)


class DiarizationService:
    """pyannote.audioとAWS Transcribeを組み合わせて話者分類を行うサービスクラス."""

    def __init__(self) -> None:
        """初期化時に各種クライアントとパイプラインを準備する."""
        # pyannote.audio パイプラインのロード
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.HUGGING_FACE_TOKEN,
        )

        # AWS クライアントの準備
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        self.transcribe_client = boto3.client(
            "transcribe",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )

        # S3バケットの存在確認・作成
        try:
            self.s3_client.head_bucket(Bucket=settings.S3_BUCKET_NAME)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.info(
                    "Bucket '%s' not found. Creating it...", settings.S3_BUCKET_NAME
                )
                location = {"LocationConstraint": settings.AWS_REGION}
                self.s3_client.create_bucket(
                    Bucket=settings.S3_BUCKET_NAME,
                    CreateBucketConfiguration=location,
                )
                logger.info("Bucket created successfully.")
            else:
                raise

    def _align_results(
        self, diarization_df: pd.DataFrame, transcribe_result: dict[str, Any]
    ) -> list[TranscriptionSegment]:
        """pyannoteとTranscribeの結果をマージする."""
        aligned_words = []
        transcribe_words = [
            item
            for item in transcribe_result["results"]["items"]
            if item["type"] == "pronunciation"
        ]

        # 各単語がどの話者の発話区間に含まれるかを判断
        for word_info in transcribe_words:
            word_start = float(word_info["start_time"])
            word_end = float(word_info["end_time"])
            word_text = word_info["alternatives"][0]["content"]

            # 単語の中心時刻が最も近い話者区間を探す
            speaker = "UNKNOWN"
            for _, row in diarization_df.iterrows():
                if row["start"] <= word_start and word_end <= row["end"]:
                    speaker = row["speaker"]
                    break

            aligned_words.append(
                {
                    "start": word_start,
                    "end": word_end,
                    "text": word_text,
                    "speaker": speaker,
                }
            )

        # 連続する同じ話者の発言を結合
        if not aligned_words:
            return []

        final_transcript = []
        current_sentence = aligned_words[0]

        for i in range(1, len(aligned_words)):
            next_word = aligned_words[i]
            if (
                next_word["speaker"] == current_sentence["speaker"]
                and (next_word["start"] - current_sentence["end"])
                < SPEECH_MERGE_THRESHOLD_S
            ):
                current_sentence["text"] += f" {next_word['text']}"
                current_sentence["end"] = next_word["end"]
            else:
                final_transcript.append(TranscriptionSegment(**current_sentence))
                current_sentence = next_word

        final_transcript.append(TranscriptionSegment(**current_sentence))
        return final_transcript

    async def _run_transcribe_job(self, s3_uri: str, job_name: str) -> dict[str, Any]:
        """AWS Transcribeジョブを実行し、結果のJSONを返す."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self.transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                LanguageCode="ja-JP",
                Media={"MediaFileUri": s3_uri},
                # 文字起こしのみが必要なため、話者分類はオフ
                Settings={"ShowSpeakerLabels": False},
            ),
        )

        start_time = time.time()
        while time.time() - start_time < JOB_TIMEOUT_S:
            job_status_response = await loop.run_in_executor(
                None,
                lambda: self.transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                ),
            )
            status = job_status_response["TranscriptionJob"]["TranscriptionJobStatus"]
            if status in ["COMPLETED", "FAILED"]:
                break
            await asyncio.sleep(POLL_INTERVAL_S)
        else:
            timeout_error_msg = (
                "Transcription job timed out after waiting for too long."
            )
            logger.error(timeout_error_msg)
            raise TimeoutError(timeout_error_msg)

        if status == "FAILED":
            reason = job_status_response["TranscriptionJob"].get(
                "FailureReason", "Unknown"
            )
            runtime_error_msg = f"Transcription job failed with reason: {reason}"
            logger.error(runtime_error_msg)
            raise RuntimeError(runtime_error_msg)

        result_uri = job_status_response["TranscriptionJob"]["Transcript"][
            "TranscriptFileUri"
        ]
        async with httpx.AsyncClient() as client:
            response = await client.get(result_uri)
            response.raise_for_status()
            return response.json()

    async def process_audio(
        self, file: UploadFile, filename: str
    ) -> DiarizationResponse:
        """音声ファイルを処理し、話者分類と文字起こしの結果をマージして返す."""
        loop = asyncio.get_running_loop()
        temp_dir = Path("temp")
        await loop.run_in_executor(
            None, lambda: temp_dir.mkdir(parents=True, exist_ok=True)
        )

        file_id = uuid.uuid4()
        temp_file_path = temp_dir / f"{file_id}{Path(filename).suffix}"
        wav_path = temp_dir / f"{file_id}.wav"
        s3_object_key = f"uploads/{wav_path.name}"
        job_name = f"diarization-{file_id}"

        try:
            # 音声ファイルを一時保存し、WAVに変換
            async with aiofiles.open(temp_file_path, "wb") as f:
                content = await file.read()
                await f.write(content)

            await loop.run_in_executor(
                None,
                lambda: AudioSegment.from_file(temp_file_path).export(
                    wav_path, format="wav"
                ),
            )

            # pyannote.audioで話者分類を実行
            diarization = await loop.run_in_executor(
                None, self.diarization_pipeline, str(wav_path)
            )
            diarization_df = pd.DataFrame(
                [
                    (turn.start, turn.end, speaker)
                    for turn, _, speaker in diarization.itertracks(yield_label=True)
                ],
                columns=["start", "end", "speaker"],
            )

            # AWS Transcribeで文字起こしを実行
            s3_uri = f"s3://{settings.S3_BUCKET_NAME}/{s3_object_key}"
            with wav_path.open("rb") as f:
                await loop.run_in_executor(
                    None,
                    lambda: self.s3_client.put_object(
                        Bucket=settings.S3_BUCKET_NAME, Key=s3_object_key, Body=f
                    ),
                )

            transcribe_result = await self._run_transcribe_job(s3_uri, job_name)

            # 結果をマージ
            transcription = self._align_results(diarization_df, transcribe_result)
            return DiarizationResponse(transcription=transcription)

        finally:
            # 一時ファイルをクリーンアップ
            with suppress(FileNotFoundError):
                await loop.run_in_executor(None, temp_file_path.unlink)
                await loop.run_in_executor(None, wav_path.unlink)
            with suppress(ClientError):
                await loop.run_in_executor(
                    None,
                    lambda: self.s3_client.delete_object(
                        Bucket=settings.S3_BUCKET_NAME, Key=s3_object_key
                    ),
                )
