use std::collections::BTreeMap;
use std::io::{IsTerminal, Read, Write};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

const SAMPLE_RATE: u32 = 16_000;

const DEFAULT_MINUTES_PROMPT: &str = include_str!("../prompts/minutes.ja.txt");

#[derive(Debug, Clone)]
struct TextSegment {
    t0: f64,
    t1: f64,
    text: String,
    no_speech: f32,
}

struct Turn {
    t0: f64,
    t1: f64,
    speaker: String,
    text: String,
}

fn audio_stream_index(path: &str) -> usize {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            path,
        ])
        .output()
        .expect("ffprobe not found");
    let idx: Vec<usize> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    match idx.as_slice() {
        [only] => *only,
        [] => panic!("no audio stream in {path}"),
        many => panic!("{} audio streams {many:?}; pick one explicitly", many.len()),
    }
}

fn decode(path: &str) -> Vec<f32> {
    let stream = audio_stream_index(path);
    let mut child = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            path,
            "-map",
            &format!("0:{stream}"),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("ffmpeg not found");
    let mut raw = Vec::new();
    child
        .stdout
        .as_mut()
        .unwrap()
        .read_to_end(&mut raw)
        .unwrap();
    assert!(child.wait().unwrap().success(), "ffmpeg failed");
    raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Decode segment bytes, carrying an incomplete trailing UTF-8 sequence into the next
/// segment. whisper splits at token boundaries, so a multi-byte character can straddle
/// two segments; decoding each one independently turns it into two replacement chars.
fn stitch(carry: &mut Vec<u8>, bytes: &[u8]) -> String {
    carry.extend_from_slice(bytes);
    let mut out = String::new();
    loop {
        match std::str::from_utf8(carry) {
            Ok(s) => {
                out.push_str(s);
                carry.clear();
                return out;
            }
            Err(e) => {
                let good = e.valid_up_to();
                out.push_str(std::str::from_utf8(&carry[..good]).unwrap());
                match e.error_len() {
                    // Truncated tail: hold it back for the next segment.
                    None => {
                        carry.drain(..good);
                        return out;
                    }
                    // Bytes that can never complete. Mark them and carry on through
                    // the rest, which is otherwise stranded until the next segment
                    // and lost outright if this was the last one.
                    Some(n) => {
                        out.push(char::REPLACEMENT_CHARACTER);
                        carry.drain(..good + n);
                    }
                }
            }
        }
    }
}

const PROGRESS_WIDTH: usize = 32;

/// A percentage bar redrawn in place on stderr. Inert when stderr is not a terminal, so
/// the carriage returns never reach a pipe or a log file.
struct Progress {
    label: &'static str,
    enabled: bool,
    shown: i32,
}

impl Progress {
    fn new(label: &'static str) -> Self {
        Self {
            label,
            enabled: std::io::stderr().is_terminal(),
            shown: -1,
        }
    }

    fn set(&mut self, percent: i32) {
        let percent = percent.clamp(0, 100);
        if !self.enabled || percent == self.shown {
            return;
        }
        self.shown = percent;
        let filled = PROGRESS_WIDTH * percent as usize / 100;
        eprint!(
            "\r{} [{}{}] {percent:>3}%",
            self.label,
            "\u{2588}".repeat(filled),
            "\u{00b7}".repeat(PROGRESS_WIDTH - filled),
        );
        let _ = std::io::stderr().flush();
    }

    /// Clears the line so the summary that follows takes its place.
    fn finish(&mut self) {
        if self.enabled {
            eprint!("\r\u{1b}[2K");
            let _ = std::io::stderr().flush();
        }
    }
}

fn transcribe(audio: &[f32], model: &str, lang: &str) -> Vec<TextSegment> {
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

    // whisper.cpp writes forty lines of device and model detail to stderr on load.
    // The hook routes them to the `log` crate, which drops them without a logger.
    whisper_rs::install_logging_hooks();

    let ctx = WhisperContext::new_with_params(model, WhisperContextParameters::default())
        .expect("failed to load whisper model");
    let mut state = ctx.create_state().expect("failed to create whisper state");

    // Greedy on purpose. Beam search and initial_prompt were both measured on real
    // meeting audio: beam search replaced whole passages with "ご視聴ありがとうございました",
    // and initial_prompt produced byte-identical output whether it held domain vocabulary
    // or unrelated nouns, so it does not steer the lexicon at all.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some(lang));

    // Whisper feeds each window's decoded tokens into the next window's prompt, so a
    // repetition loop sustains itself instead of decaying: on a 157-minute recording one
    // loop that began at 09:50 ran to the end, filling 93% of the transcript with a single
    // fabricated sentence. `no_context` does not help here -- it clears the carry-over only
    // between whisper_full calls, and this is one call per file. Zeroing n_max_text_ctx is
    // what actually skips the carry-over, at the cost of proper-noun consistency across
    // windows.
    params.set_n_max_text_ctx(0);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_n_threads(num_threads());

    let progress = Arc::new(Mutex::new(Progress::new("transcribing")));
    let sink = Arc::clone(&progress);
    params.set_progress_callback_safe(move |percent| sink.lock().unwrap().set(percent));

    state.full(params, audio).expect("whisper failed");
    progress.lock().unwrap().finish();

    let mut carry = Vec::new();
    let mut out = Vec::new();
    for i in 0..state.full_n_segments() {
        let Some(seg) = state.get_segment(i) else {
            continue;
        };
        let Ok(bytes) = seg.to_bytes() else { continue };
        let text = stitch(&mut carry, bytes);
        if text.trim().is_empty() {
            continue;
        }
        out.push(TextSegment {
            t0: seg.start_timestamp() as f64 / 100.0, // centiseconds
            t1: seg.end_timestamp() as f64 / 100.0,
            text,
            no_speech: seg.no_speech_probability(),
        });
    }

    // Anything still held back can never complete, so it is reported rather than
    // dropped on the way out.
    if !carry.is_empty() {
        eprintln!(
            "discarded {} trailing bytes that form no character",
            carry.len()
        );
    }
    out
}

fn num_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .min(8)
}

/// Diarization doubles as the speech gate: whisper hallucinates filler on long silences,
/// and reports `no_speech_probability = 0` while doing it, so a text segment overlapping
/// no speaker at all is dropped rather than attributed.
fn merge(texts: &[TextSegment], diar: &[(f64, f64, String)]) -> (Vec<Turn>, usize) {
    let mut turns: Vec<Turn> = Vec::new();
    let mut dropped = 0;

    for t in texts {
        let mut top = ("", 0.0f64);
        for (d0, d1, spk) in diar {
            let ov = t.t1.min(*d1) - t.t0.max(*d0);
            if ov > top.1 {
                top = (spk, ov);
            }
        }
        if top.1 <= 0.0 {
            dropped += 1;
            continue;
        }

        let speaker = top.0.to_string();
        let text = t.text.trim();
        if text.is_empty() {
            continue;
        }

        match turns.last_mut() {
            Some(last) if last.speaker == speaker => {
                // A space marks the segment boundary without inventing punctuation.
                last.text.push(' ');
                last.text.push_str(text);
                last.t1 = t.t1;
            }
            _ => turns.push(Turn {
                t0: t.t0,
                t1: t.t1,
                speaker,
                text: text.to_string(),
            }),
        }
    }
    (turns, dropped)
}

fn mmss(t: f64) -> String {
    format!("{:02}:{:02}", (t / 60.0) as u32, (t % 60.0) as u32)
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if let Some(arg) = std::env::args().nth(1) {
        if arg == "--version" || arg == "-V" {
            println!("koe {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
    }

    let Some(media) = std::env::args().nth(1) else {
        eprintln!("usage: koe <audio-or-video-file>");
        eprintln!();
        eprintln!("Writes <input dir>/<input name>/transcript.txt, and minutes.md when");
        eprintln!("KOE_LLM_MODEL points at a GGUF.");
        eprintln!();
        eprintln!("  KOE_WHISPER_MODEL   path to a whisper ggml model (required)");
        eprintln!("  KOE_LLM_MODEL       path to a GGUF for minutes (optional)");
        eprintln!("  KOE_LANG            whisper language, default ja");
        eprintln!("  KOE_MINUTES_PROMPT  override the built-in minutes prompt");
        std::process::exit(2);
    };
    let model = std::env::var("KOE_WHISPER_MODEL")
        .unwrap_or_else(|_| "models/ggml-large-v3-turbo.bin".into());
    if !std::path::Path::new(&model).exists() {
        eprintln!("whisper model not found: {model}");
        eprintln!("set KOE_WHISPER_MODEL, or see the README for how to fetch one.");
        std::process::exit(1);
    }
    let lang = std::env::var("KOE_LANG").unwrap_or_else(|_| "ja".into());

    eprintln!("recording consent is your responsibility; make sure participants agreed.");

    let audio = decode(&media);
    let secs = audio.len() as f64 / SAMPLE_RATE as f64;
    eprintln!("decoded {secs:.0}s");

    let t = std::time::Instant::now();
    let texts = transcribe(&audio, &model, &lang);
    eprintln!(
        "transcribed in {:.0}s -> {} segments",
        t.elapsed().as_secs_f64(),
        texts.len()
    );
    if std::env::var("KOE_DEBUG_SEGMENTS").is_ok() {
        for s in texts.iter().take(12) {
            eprintln!(
                "  {:7.2}-{:7.2} no_speech={:.3} {:?}",
                s.t0, s.t1, s.no_speech, s.text
            );
        }
        eprintln!("  ...");
        for s in texts.iter().skip(texts.len().saturating_sub(4)) {
            eprintln!(
                "  {:7.2}-{:7.2} no_speech={:.3} {:?}",
                s.t0, s.t1, s.no_speech, s.text
            );
        }
    }

    let t = std::time::Instant::now();
    let mut pipeline =
        speakrs::OwnedDiarizationPipeline::from_pretrained(speakrs::ExecutionMode::CoreMl)?;
    let result = pipeline.run(&audio)?;
    eprintln!("diarized in {:.0}s", t.elapsed().as_secs_f64());

    let mut exclusive = result.discrete_diarization.clone();
    exclusive.make_exclusive();
    let diar: Vec<(f64, f64, String)> = exclusive
        .to_segments()
        .iter()
        .map(|s| (s.start, s.end, s.speaker.to_string()))
        .collect();

    let mut totals: BTreeMap<String, f64> = BTreeMap::new();
    for (a, b, s) in &diar {
        *totals.entry(s.clone()).or_default() += b - a;
    }
    let mut ranked: Vec<_> = totals.iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    let label: BTreeMap<String, String> = ranked
        .iter()
        .enumerate()
        .map(|(i, (k, _))| ((*k).clone(), ((b'A' + i as u8) as char).to_string()))
        .collect();

    let (turns, dropped) = merge(&texts, &diar);
    if dropped > 0 {
        eprintln!("dropped {dropped} text segments with no overlapping speech");
    }

    let input = std::path::Path::new(&media);
    let stem = input.file_stem().expect("input has no file name");
    let outdir = input
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join(stem);
    std::fs::create_dir_all(&outdir)?;

    let mut out = String::new();
    out.push_str(&format!("source: {media}\n"));
    out.push_str(&format!("duration: {}\n", mmss(secs)));
    out.push_str(&format!(
        "speech: {} ({:.0}%)\n",
        mmss(totals.values().sum()),
        totals.values().sum::<f64>() / secs * 100.0
    ));
    out.push_str("speakers:");
    for (k, l) in &label {
        out.push_str(&format!(" {l}={}", mmss(totals[k])));
    }
    out.push('\n');
    out.push_str("note: speaker labels come from voice clustering, not identity.\n");
    out.push_str("note: attribution is roughly 85% accurate by duration; errors concentrate in\n");
    out.push_str(
        "      short backchannels and rapid exchanges, not in long stretches of speech.\n\n",
    );
    for turn in &turns {
        let l = label
            .get(&turn.speaker)
            .cloned()
            .unwrap_or_else(|| "?".into());
        out.push_str(&format!("[{}] {}: {}\n", mmss(turn.t0), l, turn.text));
    }

    let transcript_path = outdir.join("transcript.txt");
    std::fs::write(&transcript_path, &out)?;
    eprintln!("wrote {}", transcript_path.display());

    match write_minutes(&outdir, &out) {
        Ok(Some(path)) => eprintln!("wrote {}", path.display()),
        Ok(None) => eprintln!("skipped minutes: set KOE_LLM_MODEL to a GGUF to generate them"),
        Err(e) => eprintln!("minutes failed: {e}"),
    }
    Ok(())
}

/// Run the minutes prompt through llama.cpp. Absent a model this is skipped rather than
/// failing: the transcript is the expensive artifact and must not be lost to it.
fn write_minutes(
    outdir: &std::path::Path,
    transcript: &str,
) -> Result<Option<std::path::PathBuf>, Box<dyn std::error::Error + Send + Sync>> {
    let Ok(model) = std::env::var("KOE_LLM_MODEL") else {
        return Ok(None);
    };
    // Embedded so koe runs from any directory; override to experiment with wording.
    let head = match std::env::var("KOE_MINUTES_PROMPT") {
        Ok(path) => std::fs::read_to_string(&path)
            .map_err(|e| format!("minutes prompt not readable: {path}: {e}"))?,
        Err(_) => DEFAULT_MINUTES_PROMPT.to_string(),
    };

    let combined = outdir.join(".minutes-prompt.txt");
    std::fs::write(&combined, format!("{head}{transcript}"))?;

    let out = Command::new("llama-cli")
        .args([
            "-m",
            &model,
            "-f",
            &combined.to_string_lossy(),
            "-c",
            "32768",
            "-n",
            "3000",
            "--temp",
            "0.2",
            "-st",
            "--no-warmup",
            "-rea",
            "off",
        ])
        .stderr(Stdio::null())
        .output()?;
    let _ = std::fs::remove_file(&combined);
    if !out.status.success() {
        return Err(format!("llama-cli exited with {}", out.status).into());
    }

    // llama-cli prints a banner, the echoed prompt, then the answer; keep the answer.
    let text = String::from_utf8_lossy(&out.stdout);
    let body = match text.find("■ 日時") {
        Some(i) => {
            let tail = &text[i..];
            match tail.rfind("[ Prompt:") {
                Some(j) => &tail[..j],
                None => tail,
            }
        }
        None => return Err("llama-cli produced no minutes".into()),
    };

    let path = outdir.join("minutes.md");
    std::fs::write(&path, format!("{}\n", body.trim_end()))?;
    Ok(Some(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(t0: f64, t1: f64, text: &str) -> TextSegment {
        TextSegment {
            t0,
            t1,
            text: text.into(),
            no_speech: 0.0,
        }
    }

    #[test]
    fn stitch_passes_through_complete_input() {
        let mut carry = Vec::new();
        assert_eq!(stitch(&mut carry, "おはよう".as_bytes()), "おはよう");
        assert!(carry.is_empty());
    }

    #[test]
    fn stitch_rejoins_a_character_split_across_segments() {
        let whole = "あい".as_bytes();
        let (head, tail) = whole.split_at(4); // mid-way through the second character
        let mut carry = Vec::new();

        assert_eq!(stitch(&mut carry, head), "あ");
        assert!(!carry.is_empty(), "the truncated tail must be held back");
        assert_eq!(stitch(&mut carry, tail), "い");
        assert!(carry.is_empty());
    }

    #[test]
    fn stitch_handles_a_segment_that_opens_with_a_continuation_byte() {
        let whole = "日本語".as_bytes();
        let mut carry = Vec::new();
        let mut out = String::new();
        // One byte at a time is the worst case: every boundary lands mid-character.
        for b in whole {
            out.push_str(&stitch(&mut carry, &[*b]));
        }
        assert_eq!(out, "日本語");
        assert!(carry.is_empty());
    }

    #[test]
    fn stitch_drops_bytes_that_can_never_complete() {
        let mut carry = Vec::new();
        // 0xFF starts no valid sequence. It is marked, and the byte after it still
        // comes through in the same call.
        assert_eq!(stitch(&mut carry, &[0xFF, b'a']), "\u{FFFD}a");
        assert!(carry.is_empty());
    }

    #[test]
    fn stitch_keeps_nul_which_is_valid_utf8() {
        let mut carry = Vec::new();
        assert_eq!(stitch(&mut carry, &[b'a', 0x00, b'b']), "a\0b");
    }

    #[test]
    fn merge_attributes_each_segment_to_its_dominant_speaker() {
        let texts = vec![seg(0.0, 2.0, "ひとつめ"), seg(3.0, 5.0, "ふたつめ")];
        let diar = vec![
            (0.0, 2.0, "SPEAKER_00".to_string()),
            (3.0, 5.0, "SPEAKER_01".to_string()),
        ];
        let (turns, dropped) = merge(&texts, &diar);

        assert_eq!(dropped, 0);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker, "SPEAKER_00");
        assert_eq!(turns[1].speaker, "SPEAKER_01");
    }

    #[test]
    fn merge_collapses_a_run_of_one_speaker_into_a_single_turn() {
        let texts = vec![seg(0.0, 1.0, "まず"), seg(1.0, 2.0, "つぎ")];
        let diar = vec![(0.0, 2.0, "SPEAKER_00".to_string())];
        let (turns, _) = merge(&texts, &diar);

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].text, "まず つぎ");
        assert_eq!(turns[0].t1, 2.0);
    }

    #[test]
    fn merge_drops_text_that_overlaps_no_speech() {
        // Whisper's hallucinated filler over a silence lands outside every speaker.
        let texts = vec![seg(0.0, 5.0, "はい"), seg(10.0, 11.0, "本題")];
        let diar = vec![(10.0, 11.0, "SPEAKER_00".to_string())];
        let (turns, dropped) = merge(&texts, &diar);

        assert_eq!(dropped, 1);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].text, "本題");
    }
}
