class Koe < Formula
  desc "Speaker-attributed transcripts and minutes for Japanese meetings, entirely on your machine"
  homepage "https://github.com/hiroaki222/koe"
  url "https://github.com/hiroaki222/koe/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license any_of: ["MIT", "Apache-2.0"]
  head "https://github.com/hiroaki222/koe.git", branch: "main"

  # whisper.cpp is compiled from source by whisper-rs.
  depends_on "cmake" => :build
  depends_on "rust" => :build
  # koe shells out to both rather than linking a decoder.
  depends_on "ffmpeg"
  # Metal and CoreML are not optional here; there is no CPU fallback worth shipping.
  depends_on arch: :arm64
  depends_on :macos

  def install
    system "cargo", "install", *std_cargo_args
  end

  def caveats
    <<~EOS
      koe needs a Whisper model. It is not bundled: it is 1.5 GB and changes
      independently of this formula.

        mkdir -p ~/.koe/models
        curl -L -o ~/.koe/models/ggml-large-v3-turbo.bin \\
          https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
        export KOE_WHISPER_MODEL=~/.koe/models/ggml-large-v3-turbo.bin

      Minutes are optional. To generate them, install llama.cpp, fetch a GGUF
      that handles Japanese well, and point KOE_LLM_MODEL at it:

        brew install llama.cpp
        export KOE_LLM_MODEL=/path/to/model.gguf
    EOS
  end

  test do
    # No model is present in the test sandbox, so the run cannot reach whisper.
    # What is worth asserting is that the binary starts and reports its contract.
    assert_match "usage: koe", shell_output("#{bin}/koe 2>&1", 2)
  end
end
