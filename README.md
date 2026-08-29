# koe

日本語の会議録音から、誰が何を話したかを起こして議事録にするツールです。すべて手元のマシンで動き、音声もテキストも外に出ません。

> **録音の前に参加者の同意を取ってください。** ツールは代われません。

```
koe meeting.mp4
  → meeting/transcript.txt   話者ラベルとタイムスタンプ付きの発話記録
  → meeting/minutes.md       決定事項とToDo、議題
```

Apple Silicon 専用です。文字起こしは Metal、話者分離は CoreML で動きます。

## 日本語専用

Whisper の言語を `ja` に固定しています。自動判定は冒頭 30 秒しか見ないので、無音や短い挨拶で始まる会議は英語として文字起こしされます。翻訳モードも同じ理由で切ってあります。議事録のプロンプトも日本語です。

## インストール

```sh
brew install hiroaki222/tap/koe
```

自分でビルドするなら `cmake` が要ります。whisper.cpp をコンパイルするためです。

```sh
brew install cmake ffmpeg
cargo build --release
```

## モデル

モデルは同梱していません。サイズが大きく、このツールとは別に更新されるためです。

```sh
mkdir -p ~/.koe/models
curl -L -o ~/.koe/models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

議事録は任意です。生成するにはローカルの言語モデルが要ります。

```sh
brew install llama.cpp
# 日本語を扱える instruction-tuned な GGUF を用意してください
```

## 使い方

```sh
export KOE_WHISPER_MODEL=~/.koe/models/ggml-large-v3-turbo.bin
export KOE_LLM_MODEL=~/.koe/models/<用意したモデル>.gguf   # 任意

koe ~/recordings/meeting.mp4
```

ffmpeg が読める形式なら何でも入ります。mp4、m4a、wav、mov、mp3 など。音声ストリームが複数ある動画は、推測せずにエラーで止まります。

実行するとこう出ます。

```
recording consent is your responsibility; make sure participants agreed.
decoded 1930s
transcribed in 42s -> 186 segments
diarized in 4s
dropped 12 text segments with no overlapping speech
wrote /Users/you/recordings/meeting/transcript.txt
wrote /Users/you/recordings/meeting/minutes.md
```

`dropped` は、どの話者とも重ならなかったテキストの数です。無音区間に Whisper が捏造した相槌がここに入ります。

出力は入力の隣にディレクトリを作って置きます。

```
~/recordings/
├── meeting.mp4
└── meeting/
    ├── transcript.txt
    └── minutes.md
```

`KOE_LLM_MODEL` を設定しなければ文字起こしだけが出ます。文字起こしには数分かかるので、言語モデルが無かったり落ちたりしても、それを巻き添えにはしません。

### 処理時間

M4 Pro での実測です。ほとんどが文字起こしで、話者分離は誤差の範囲です。

| 会議の長さ | 文字起こし | 話者分離 | 議事録 (27B) |
|---|---|---|---|
| 20 分 | 28 秒 | 3 秒 | 1 分 15 秒 |
| 2 分 | 6 秒 | 1 秒 | 20 秒 |

### 議事録のプロンプトを差し替える

組み込みのプロンプトはバイナリに埋め込んであります。書き換えて試すには、ファイルを渡します。

```sh
KOE_MINUTES_PROMPT=./my-prompt.txt koe meeting.mp4
```

プロンプトの末尾に文字起こしがそのまま連結されるので、フォーマットの指示はファイルの中に書ききってください。組み込みのものは [prompts/minutes.ja.txt](prompts/minutes.ja.txt) にあります。

### 環境変数

| 環境変数 | 意味 |
|---|---|
| `KOE_WHISPER_MODEL` | Whisper の ggml モデル。既定は `models/ggml-large-v3-turbo.bin` |
| `KOE_LLM_MODEL` | 議事録に使う GGUF。未設定なら議事録を作りません |
| `KOE_LANG` | Whisper の言語。既定は `ja` |
| `KOE_MINUTES_PROMPT` | 組み込みの議事録プロンプトを差し替えるファイル |

## 文字起こしの形式

```
source: meeting.mp4
duration: 32:10
speech: 21:40 (67%)
speakers: A=12:30 B=09:10
note: speaker labels come from voice clustering, not identity.
note: attribution is roughly 85% accurate by duration; errors concentrate in
      short backchannels and rapid exchanges, not in long stretches of speech.

[03:12] A: 来週のリリースなんですけど テスト環境の準備って どれくらいかかりそうですか
[03:20] B: 二日あれば 一日で終わるかもしれないですけど 余裕を見て二日で
```

言語モデルが読むための形式です。ターンごとの見出しを持たず、タイムスタンプは分単位で、句読点を捏造せずセグメントの切れ目をスペースで表します。

## 仕組み

```
音声・動画ファイル
  ├─ ffprobe        音声ストリームを選ぶ。複数あるときは推測せず拒否する
  ├─ ffmpeg         16 kHz モノラル f32 に正規化
  ├─ whisper-rs     タイムスタンプ付きのテキスト
  ├─ speakrs        誰がいつ話したか
  ├─ merge          各テキストを最も重なる話者に割り当てる
  └─ llama-cli      モデルが設定されていれば議事録
```

**話者分離を発話の判定にも使っています。** Whisper は長い無音に相槌を捏造します。冒頭 15 分が無音の録音では `はい` が延々と並びました。しかもその間ずっと `no_speech_probability` は 0 を返すので、この値では弾けません。どの話者とも重ならないテキストを捨てる、という形に落ち着きました。

セグメントのテキストは繰り越しバッファでデコードしています。Whisper はトークン境界で区切るため、マルチバイト文字が 2 つのセグメントにまたがることがあります。独立してデコードすると、その文字は置換文字 2 個に化けます。

議事録のプロンプトは、ほとんどが禁止事項です。テンプレートを渡された言語モデルは空欄を埋めようとして、日付や担当、期限を捏造します。該当が無い節は空のままにすること、判断のつかない語は推測も省略もせず印を付けて残すこと、月や数値を会議全体と照らして検算すること。

## 限界

- 話者の帰属は 2 人の通話で時間換算 85% 程度です。長く話している区間は信頼できますが、短い相槌や割り込みは外します。
- Whisper がここでは句読点を出しません。文字起こしは切れ目のない文になります。
- 同音異義語の誤変換は議事録にも残ります。不確かな語に印を付けるようプロンプトで指示していますが、モデルがそれに従うかは安定しません。
- 同時発話は捨てています。どの瞬間も 1 人の話者に割り当てます。
