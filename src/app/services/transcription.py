import asyncio
import logging
import time
import uuid
from contextlib import suppress
from typing import Any

import boto3
import httpx
from botocore.exceptions import ClientError
from fastapi import UploadFile

from src.app.core.settings import settings
from src.app.models.diarization import DiarizationResponse, TranscriptionSegment

# ロガーのセットアップ
logger = logging.getLogger(__name__)

SPEECH_MERGE_THRESHOLD_S = 0.5  # 同一話者の発言を結合する閾値(秒)
POLL_INTERVAL_S = 5  # Transcribeジョブのステータスを確認する間隔(秒)
JOB_TIMEOUT_S = 600  # Transcribeジョブのタイムアウト(秒)


class DiarizationService:
    """AWS Transcribeを使用して話者分類と文字起こしを行うサービスクラス."""

    def __init__(self) -> None:
        """初期化時にboto3クライアントを準備する."""
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

        try:
            self.s3_client.head_bucket(Bucket=settings.S3_BUCKET_NAME)
        except ClientError as e:
            # バケットが存在しない場合 (404 Not Found)
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
                # その他のClientErrorは再送出
                raise

    def _parse_transcribe_result(
        self, result: dict[str, Any]
    ) -> list[TranscriptionSegment]:
        """AWS Transcribeの結果JSONをパースして整形する."""
        transcripts = []
        speaker_map = {}

        # Transcribeの結果から、各発言セグメントの情報を取得
        for segment in result["results"]["speaker_labels"]["segments"]:
            for item in segment["items"]:
                speaker_label = item["speaker_label"]
                if speaker_label not in speaker_map:
                    # SPEAKER_00, SPEAKER_01... の形式にマッピング
                    speaker_map[speaker_label] = f"SPEAKER_{len(speaker_map):02d}"

        items = result["results"]["items"]
        current_segment = None

        for item in items:
            if item["type"] == "pronunciation":
                start_time = float(item["start_time"])
                end_time = float(item["end_time"])
                content = item["alternatives"][0]["content"]
                speaker_label = item.get("speaker_label", "spk_0")
                speaker = speaker_map.get(speaker_label, "UNKNOWN")

                if (
                    current_segment
                    and current_segment["speaker"] == speaker
                    and (start_time - current_segment["end"]) < SPEECH_MERGE_THRESHOLD_S
                ):
                    current_segment["text"] += f" {content}"
                    current_segment["end"] = end_time
                else:
                    if current_segment:
                        transcripts.append(TranscriptionSegment(**current_segment))
                    current_segment = {
                        "speaker": speaker,
                        "text": content,
                        "start": start_time,
                        "end": end_time,
                    }

        if current_segment:
            transcripts.append(TranscriptionSegment(**current_segment))

        return transcripts

    async def process_audio(
        self, file: UploadFile, filename: str
    ) -> DiarizationResponse:
        """音声ファイルをS3にアップロードし、Transcribeジョブを実行して結果を返す."""
        job_name = f"diarization-{uuid.uuid4()}"
        s3_object_key = f"uploads/{job_name}-{filename}"
        loop = asyncio.get_running_loop()

        try:
            audio_content = await file.read()
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.put_object(
                    Bucket=settings.S3_BUCKET_NAME,
                    Key=s3_object_key,
                    Body=audio_content,
                ),
            )
            s3_uri = f"s3://{settings.S3_BUCKET_NAME}/{s3_object_key}"

            await loop.run_in_executor(
                None,
                lambda: self.transcribe_client.start_transcription_job(
                    TranscriptionJobName=job_name,
                    LanguageCode="ja-JP",
                    Media={"MediaFileUri": s3_uri},
                    Settings={"ShowSpeakerLabels": True, "MaxSpeakerLabels": 5},
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
                job_status = job_status_response["TranscriptionJob"][
                    "TranscriptionJobStatus"
                ]

                if job_status in ["COMPLETED", "FAILED"]:
                    break
                await asyncio.sleep(POLL_INTERVAL_S)
            else:
                timeout_message = "Transcription job timed out."
                raise TimeoutError(timeout_message)

            if job_status == "FAILED":
                reason = job_status_response["TranscriptionJob"].get(
                    "FailureReason", "Unknown reason"
                )
                error_message = f"Transcription job failed: {reason}"
                raise RuntimeError(error_message)

            # 4. 結果JSONを取得してパース
            result_uri = job_status_response["TranscriptionJob"]["Transcript"][
                "TranscriptFileUri"
            ]
            async with httpx.AsyncClient() as client:
                response = await client.get(result_uri)
                response.raise_for_status()
                result_json = response.json()

            transcription = self._parse_transcribe_result(result_json)
            return DiarizationResponse(transcription=transcription)

        finally:
            with suppress(ClientError):
                await loop.run_in_executor(
                    None,
                    lambda: self.s3_client.delete_object(
                        Bucket=settings.S3_BUCKET_NAME, Key=s3_object_key
                    ),
                )
