# Speaker Diarization API

[](https://github.com/astral-sh/ruff)

`pyannote.audio` と `AWS Transcribe` を使用した、話者分類と文字起こしを行うFastAPIアプリケーションです。

## 概要

このAPIは、アップロードされた音声ファイル（mp4, mp3, wav形式に対応）を受け取り、話者ごとに発言を分割し、それぞれの発言を文字起こしします。話者分類には`pyannote.audio`、文字起こしには`AWS Transcribe`を利用しています。

-----

## 1\. 事前準備

### 1.1. 必要なもの

  - [git](https://git-scm.com/)
  - [Docker](https://www.docker.com/ja-jp/)
  - [Dev Containers (VSCode拡張機能)](https://code.visualstudio.com/docs/devcontainers/containers)

### 1.2. 環境変数の設定

プロジェクトのルートディレクトリに`.env`ファイルを作成し、各種キーを設定してください。`.env.sample`をコピーして使用できます。

```bash
cp .env.sample .env
```

**.envファイルに必要な変数:**

  - `AWS_ACCESS_KEY_ID`: AWSのアクセスキーID
  - `AWS_SECRET_ACCESS_KEY`: AWSのシークレットアクセスキー
  - `AWS_REGION`: (任意) AWSリージョン（デフォルト: `ap-northeast-1`）
  - `S3_BUCKET_NAME`: `AWS Transcribe`が使用するS3バケット名
  - `HUGGING_FACE_TOKEN`: Hugging Faceのアクセストークン（次のセクションで取得します）

### 1.3. Hugging Faceの認証設定

`pyannote.audio`のモデルを使用するために、Hugging Faceの認証設定が必要です。

1.  **アカウント作成とモデルへの同意**

      - [Hugging Face公式サイト](https://huggingface.co/)でアカウントを作成します。
      - ログイン後、以下のモデルページにアクセスし、利用規約に同意 (**Agree and access repository**) してください。
          - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
          - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

2.  **アクセストークンの作成**

      - [こちらのページ](https://huggingface.co/settings/tokens) (Settings \> Access Tokens) から、`Role`が **`read`** の新しいアクセストークンを作成します。
      - 作成したトークンをコピーし、ステップ1.2で作成した`.env`ファイルの`HUGGING_FACE_TOKEN`に貼り付けてください。

-----

## 2\. FastAPIサーバの起動

このプロジェクトでは、`hatch`を使用してDockerコンテナのビルドと起動を管理します。

### 2.1. Dockerイメージのビルド

以下のコマンドを実行して、FastAPIアプリケーションのDockerイメージをビルドします。

```bash
hatch run build-server
```

### 2.2. Dockerコンテナの起動

ビルドが完了したら、以下のコマンドでDockerコンテナを起動します。

```bash
hatch run run-server
```

サーバーが正常に起動すると、以下のメッセージが表示されます。

```
🚀 FastAPI server is running at http://localhost:8000
📝 Swagger UI is available at http://localhost:8000/docs
```

  - **APIサーバー**: `http://localhost:8000`
  - **Swagger UI (APIドキュメント)**: `http://localhost:8000/docs`

-----

## 3\. APIエンドポイント

### `POST /api/v1/diarize`

音声ファイルをアップロードして、話者分類と文字起こしを実行します。

**リクエストボディ:**

  - `file`: 音声ファイル (mp4, mp3, wav)

**レスポンス:**

話者ごとの発言セグメントのリストを返します。

```json
{
  "transcription": [
    {
      "speaker": "SPEAKER_00",
      "text": "こんにちは、これはテストです。",
      "start": 0.5,
      "end": 2.5
    },
    {
      "speaker": "SPEAKER_01",
      "text": "次の発言です。",
      "start": 3.0,
      "end": 4.5
    }
  ]
}
```
