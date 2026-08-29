> [!IMPORTANT]
> ### 使う前に
>
> koe は Whisper のモデルを必要とします。1.5 GB あり、このツールとは別に更新されるため同梱していません。
>
> ```sh
> mkdir -p ~/.koe/models
> curl -L -o ~/.koe/models/ggml-large-v3-turbo.bin \
>   https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
> export KOE_WHISPER_MODEL=~/.koe/models/ggml-large-v3-turbo.bin
> ```
>
> 議事録まで出すには `llama.cpp` と、日本語を扱える GGUF が要ります。設定しなければ文字起こしだけが出ます。
>
> ```sh
> brew install llama.cpp
> export KOE_LLM_MODEL=/path/to/model.gguf
> ```
>
> Apple Silicon 専用です。`ffmpeg` と `ffprobe` が PATH に必要です。
