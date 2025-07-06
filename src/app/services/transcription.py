import asyncio
import uuid
from pathlib import Path
from typing import Any

import aiofiles
import pandas as pd
from fastapi import UploadFile
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment

from src.app.core.settings import settings
from src.app.models.diarization import DiarizationResponse, TranscriptionSegment

SPEECH_MERGE_THRESHOLD_S = 0.5  # 同じ話者の発言を結合する最大の間隔(秒)


class DiarizationService:
    """話者分類と文字起こし(Whisper)を行うサービスクラス."""

    def __init__(self) -> None:
        """初期化時にWhisperとpyannoteのパイプラインを準備する."""
        # 参考:https://github.com/openai/whisper
        self.whisper_model = WhisperModel(
            "large-v3",
            device="cpu",
            compute_type="int8",
        )

        # pyannote.audioパイプラインのロード
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.HUGGING_FACE_TOKEN,
        )

    def _align_whisper_and_pyannote(
        self,
        whisper_result: list[dict[str, Any]],
        diarization_result: pd.DataFrame,
    ) -> list[TranscriptionSegment]:
        """Whisperの単語とpyannoteの話者区間をマージする."""
        aligned_results = []
        for segment in whisper_result:
            # Whisperの各単語のタイムスタンプを取得
            for word in segment.get("words", []):
                word_start, word_end = word["start"], word["end"]
                # この単語が含まれる話者区間を探す
                speaker = "UNKNOWN"
                for _, row in diarization_result.iterrows():
                    if row["start"] <= word_start and word_end <= row["end"]:
                        speaker = row["speaker"]
                        break
                # 単語と話者を紐付け
                aligned_results.append(
                    {
                        "start": word_start,
                        "end": word_end,
                        "text": word["word"],
                        "speaker": speaker,
                    },
                )

        # 連続する同じ話者の発言を結合する
        if not aligned_results:
            return []

        final_transcript = []
        current_sentence_data = {
            "speaker": aligned_results[0]["speaker"],
            "text": aligned_results[0]["text"],
            "start": aligned_results[0]["start"],
            "end": aligned_results[0]["end"],
        }

        for i in range(1, len(aligned_results)):
            word_info = aligned_results[i]
            if (
                word_info["speaker"] == current_sentence_data["speaker"]
                and (word_info["start"] - current_sentence_data["end"])
                < SPEECH_MERGE_THRESHOLD_S
            ):
                current_sentence_data["text"] += word_info["text"]
                current_sentence_data["end"] = word_info["end"]
            else:
                final_transcript.append(TranscriptionSegment(**current_sentence_data))
                current_sentence_data = {
                    "speaker": word_info["speaker"],
                    "text": word_info["text"],
                    "start": word_info["start"],
                    "end": word_info["end"],
                }

        final_transcript.append(TranscriptionSegment(**current_sentence_data))

        return final_transcript

    async def process_audio(
        self, file: UploadFile, filename: str
    ) -> DiarizationResponse:
        """音声ファイルを処理し、話者分類と文字起こしの結果を返す."""
        loop = asyncio.get_running_loop()

        temp_dir = Path("temp")
        # キーワード引数を使うようにlambdaでラップ
        await loop.run_in_executor(None, lambda: temp_dir.mkdir(exist_ok=True))

        file_extension = Path(filename).suffix
        temp_file_path = temp_dir / f"{uuid.uuid4()}{file_extension}"
        wav_path = temp_dir / f"{temp_file_path.stem}.wav"

        async with aiofiles.open(temp_file_path, "wb") as buffer:
            content = await file.read()
            await buffer.write(content)

        # pydubを使って音声ファイルをwav形式に変換
        def convert_to_wav() -> None:
            audio = AudioSegment.from_file(temp_file_path)
            audio.export(wav_path, format="wav")

        await loop.run_in_executor(None, convert_to_wav)

        # pyannoteの話者分類を実行
        diarization = await loop.run_in_executor(
            None,
            self.diarization_pipeline,
            str(wav_path),
        )
        diarization_df = pd.DataFrame(
            [
                (turn.start, turn.end, speaker)
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ],
            columns=["start", "end", "speaker"],
        )

        # Whisperで文字起こしを実行
        whisper_segments, _ = await loop.run_in_executor(
            None,
            lambda: self.whisper_model.transcribe(str(wav_path), word_timestamps=True),
        )

        whisper_result_list = [seg._asdict() for seg in whisper_segments]

        transcription = self._align_whisper_and_pyannote(
            whisper_result_list,
            diarization_df,
        )

        await loop.run_in_executor(None, lambda: temp_file_path.unlink(missing_ok=True))
        await loop.run_in_executor(None, lambda: wav_path.unlink(missing_ok=True))

        return DiarizationResponse(transcription=transcription)
