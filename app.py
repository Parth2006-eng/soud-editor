import streamlit as st
import subprocess
import sys
import tempfile
import os
from pathlib import Path
import numpy as np
import librosa
from pydub import AudioSegment

st.set_page_config(page_title="AI Instrument Swapper", layout="centered")
st.title("🎚️ AI Audio Separator & Instrument Swapper")
st.write(
    "Upload a song, split it into stems with Demucs, then replace or convert "
    "any stem into a different instrument — for the whole track or just a "
    "selected section."
)

STEM_NAMES = ["vocals", "drums", "bass", "other"]

# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------
if "work_dir" not in st.session_state:
    st.session_state.work_dir = None
if "original_stems" not in st.session_state:
    st.session_state.original_stems = {}   # never overwritten, used for analysis
if "stems" not in st.session_state:
    st.session_state.stems = {}            # current/working versions used in final mix


# ---------------------------------------------------------------------------
# audio <-> numpy helpers
# ---------------------------------------------------------------------------
def synth_note(freq, duration_ms, sr=44100, amp=0.6):
    """Simple 'plucked' tone: a few harmonics with an exponential decay envelope."""
    n = max(1, int(sr * duration_ms / 1000))
    t = np.linspace(0, duration_ms / 1000, n, endpoint=False)
    wave = (
        1.00 * np.sin(2 * np.pi * freq * t)
        + 0.50 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.25 * np.sin(2 * np.pi * freq * 3 * t)
    )
    envelope = np.exp(-3.5 * t / max(duration_ms / 1000, 1e-6))
    attack = int(0.005 * sr)
    if 0 < attack < n:
        envelope[:attack] *= np.linspace(0, 1, attack)
    wave = wave * envelope
    peak = np.max(np.abs(wave)) + 1e-9
    return (wave / peak * amp).astype(np.float32)


def render_from_sample(sample_y, sample_sr, target_sr, freq, sample_base_freq, dur_ms, amp):
    y = sample_y
    if sample_sr != target_sr:
        y = librosa.resample(y=y, orig_sr=sample_sr, target_sr=target_sr)
    if sample_base_freq and sample_base_freq > 0 and freq:
        semitones = 12 * np.log2(freq / sample_base_freq)
        semitones = float(np.clip(semitones, -24, 24))
        y = librosa.effects.pitch_shift(y=y, sr=target_sr, n_steps=semitones)
    n_samples = max(1, int(target_sr * dur_ms / 1000))
    if len(y) >= n_samples:
        y = y[:n_samples].copy()
    else:
        y = np.pad(y, (0, n_samples - len(y)))
    fade_len = min(300, len(y) // 4)
    if fade_len > 0:
        y = y.astype(np.float32)
        y[-fade_len:] *= np.linspace(1, 0, fade_len)
    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak * amp).astype(np.float32)


def get_local_rms(y, sr, t, window=0.08):
    i0 = max(0, int((t - window / 2) * sr))
    i1 = min(len(y), int((t + window / 2) * sr))
    if i1 <= i0:
        return 0.0
    return float(np.sqrt(np.mean(y[i0:i1] ** 2)))


def detect_onsets(y, sr, start_s, end_s):
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    return onsets[(onsets >= start_s) & (onsets < end_s)]


def detect_pitch_track(y, sr):
    f0, _, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    times = librosa.times_like(f0, sr=sr)
    return times, f0


def resynthesize_segment(
    y, sr, start_s, end_s, mode, source,
    sample_y=None, sample_sr=None, sample_base_freq=None,
    base_note_freq=220.0, max_note_ms=350,
):
    """Returns a mono float32 array covering [start_s, end_s) built from
    detected onsets (and, in pitch mode, detected pitch) of the ORIGINAL
    stem audio, rendered using either an uploaded sample or a synthesized
    tone."""
    seg_len = max(1, int((end_s - start_s) * sr))
    out = np.zeros(seg_len, dtype=np.float32)

    onset_times = detect_onsets(y, sr, start_s, end_s)
    if len(onset_times) == 0:
        return out

    times = f0 = None
    if mode == "pitch":
        times, f0 = detect_pitch_track(y, sr)

    for i, onset_t in enumerate(onset_times):
        next_t = onset_times[i + 1] if i + 1 < len(onset_times) else end_s
        dur_ms = min(max_note_ms, max(50, (next_t - onset_t) * 1000))
        amp = float(np.clip(get_local_rms(y, sr, onset_t) * 4, 0.05, 1.0))

        freq = base_note_freq
        if mode == "pitch" and f0 is not None:
            idx = int(np.argmin(np.abs(times - onset_t)))
            if idx < len(f0) and f0[idx] and not np.isnan(f0[idx]):
                freq = float(f0[idx])

        if source == "sample" and sample_y is not None:
            note_wave = render_from_sample(
                sample_y, sample_sr, sr, freq, sample_base_freq, dur_ms, amp
            )
        else:
            note_wave = synth_note(freq, dur_ms, sr=sr, amp=amp)

        start_idx = int((onset_t - start_s) * sr)
        end_idx = min(seg_len, start_idx + len(note_wave))
        length = end_idx - start_idx
        if length > 0:
            out[start_idx:end_idx] += note_wave[:length]

    return out


def np_mono_to_segment(y, sr):
    y = np.clip(y, -1.0, 1.0)
    y_int16 = (y * 32767).astype(np.int16)
    return AudioSegment(y_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)


def splice_segment(original_path, new_mono_np, new_sr, start_s, end_s):
    """Replace [start_s, end_s) of the stem at original_path with new_mono_np,
    matching channel count / sample width / frame rate, and return the full
    resulting AudioSegment."""
    orig_seg = AudioSegment.from_file(original_path)
    start_ms, end_ms = int(start_s * 1000), int(end_s * 1000)

    before = orig_seg[:start_ms]
    after = orig_seg[end_ms:]

    new_seg = np_mono_to_segment(new_mono_np, new_sr)
    new_seg = new_seg.set_channels(orig_seg.channels)
    new_seg = new_seg.set_frame_rate(orig_seg.frame_rate)
    new_seg = new_seg.set_sample_width(orig_seg.sample_width)

    return before + new_seg + after


def note_options():
    notes = []
    for octave in range(2, 6):
        for name in ["C", "D", "E", "F", "G", "A", "B"]:
            note = f"{name}{octave}"
            notes.append((note, librosa.note_to_hz(note)))
    return notes


# ---------------------------------------------------------------------------
# 1. Upload + separate
# ---------------------------------------------------------------------------
uploaded_audio = st.file_uploader(
    "1. Upload your song", type=["mp3", "wav", "flac", "ogg", "m4a"]
)

model_choice = st.selectbox(
    "Model",
    ["htdemucs (default, fast, 4-stem)", "htdemucs_ft (fine-tuned, slower, higher quality)"],
    index=0,
)
model_name = "htdemucs_ft" if "ft" in model_choice else "htdemucs"

if uploaded_audio is not None:
    if st.button("2. Separate audio into stems"):
        with st.spinner("Running Demucs separation... this can take a minute or two"):
            work_dir = tempfile.mkdtemp()
            input_path = os.path.join(work_dir, uploaded_audio.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())

            out_dir = os.path.join(work_dir, "separated")
            cmd = [sys.executable, "-m", "demucs", "-n", model_name, "-o", out_dir, input_path]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                st.error("Separation failed. See details below.")
                st.code(result.stderr)
            else:
                track_name = Path(input_path).stem
                stem_dir = os.path.join(out_dir, model_name, track_name)
                stems = {}
                for stem in STEM_NAMES:
                    p = os.path.join(stem_dir, f"{stem}.wav")
                    if os.path.exists(p):
                        stems[stem] = p
                st.session_state.work_dir = work_dir
                st.session_state.original_stems = dict(stems)
                st.session_state.stems = dict(stems)
                st.success("Separation complete!")

# ---------------------------------------------------------------------------
# 3. Preview
# ---------------------------------------------------------------------------
if st.session_state.stems:
    st.header("3. Preview stems")
    for stem, path in st.session_state.stems.items():
        modified = path != st.session_state.original_stems[stem]
        label = f"**{stem.capitalize()}**" + (" _(modified)_" if modified else "")
        st.write(label)
        st.audio(path)

    # -----------------------------------------------------------------
    # 4. Edit a stem
    # -----------------------------------------------------------------
    st.header("4. Edit a stem")
    stem_to_edit = st.selectbox("Stem to edit", list(st.session_state.stems.keys()))

    orig_path = st.session_state.original_stems[stem_to_edit]
    stem_duration = len(AudioSegment.from_file(orig_path)) / 1000.0

    time_range = st.slider(
        "Time range to modify (seconds)",
        0.0, float(stem_duration), (0.0, float(stem_duration)), step=0.1,
    )
    start_s, end_s = time_range

    edit_mode = st.radio(
        "Edit mode",
        ["Full replacement (upload your own audio for this range)",
         "Instrument conversion (AI resynthesis)"],
    )

    # ---------------- Full replacement ----------------
    if edit_mode.startswith("Full replacement"):
        replacement_file = st.file_uploader(
            "Upload replacement audio", type=["mp3", "wav", "flac", "ogg", "m4a"], key="repl_full"
        )
        if replacement_file is not None and st.button("Apply replacement"):
            with st.spinner("Applying..."):
                work_dir = st.session_state.work_dir
                repl_path = os.path.join(work_dir, "repl_" + replacement_file.name)
                with open(repl_path, "wb") as f:
                    f.write(replacement_file.getbuffer())

                orig_seg = AudioSegment.from_file(orig_path)
                target_len_ms = int((end_s - start_s) * 1000)

                repl_audio = AudioSegment.from_file(repl_path)
                repl_audio = repl_audio.set_frame_rate(orig_seg.frame_rate)
                repl_audio = repl_audio.set_channels(orig_seg.channels)
                if len(repl_audio) > target_len_ms:
                    repl_audio = repl_audio[:target_len_ms]
                else:
                    pad = AudioSegment.silent(
                        duration=target_len_ms - len(repl_audio), frame_rate=repl_audio.frame_rate
                    )
                    repl_audio = repl_audio + pad

                before = orig_seg[: int(start_s * 1000)]
                after = orig_seg[int(end_s * 1000):]
                combined = before + repl_audio + after

                out_path = os.path.join(work_dir, f"{stem_to_edit}_modified.wav")
                combined.export(out_path, format="wav")
                st.session_state.stems[stem_to_edit] = out_path
                st.success("Stem updated.")
                st.rerun()

    # ---------------- Instrument conversion ----------------
    else:
        detect_mode_label = st.selectbox(
            "Detection mode",
            ["Auto (rhythm for drums, pitch otherwise)", "Rhythm (percussive hits)", "Pitch-tracked (melodic)"],
        )
        if detect_mode_label.startswith("Auto"):
            detect_mode = "rhythm" if stem_to_edit == "drums" else "pitch"
        elif detect_mode_label.startswith("Rhythm"):
            detect_mode = "rhythm"
        else:
            detect_mode = "pitch"

        source_choice = st.radio("Target instrument sound", ["Upload a sample", "Synthesized tone"])

        sample_y = sample_sr = sample_base_freq = None
        base_note_freq = 220.0
        max_note_ms = st.slider("Max note length (ms)", 50, 1000, 350, step=10)

        if source_choice == "Upload a sample":
            sample_file = st.file_uploader(
                "Upload a short one-shot sample of the target instrument (e.g. a single guitar pluck)",
                type=["mp3", "wav", "flac", "ogg", "m4a"], key="sample_upload",
            )
            override_freq = st.number_input(
                "Override detected sample pitch (Hz, leave 0 to auto-detect)", min_value=0.0, value=0.0, step=1.0
            )
            if sample_file is not None:
                work_dir = st.session_state.work_dir
                sample_path = os.path.join(work_dir, "sample_" + sample_file.name)
                with open(sample_path, "wb") as f:
                    f.write(sample_file.getbuffer())
                sample_y, sample_sr = librosa.load(sample_path, sr=None, mono=True)
                if override_freq > 0:
                    sample_base_freq = override_freq
                else:
                    try:
                        f0, _, _ = librosa.pyin(
                            sample_y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sample_sr
                        )
                        sample_base_freq = float(np.nanmedian(f0)) if np.any(~np.isnan(f0)) else 220.0
                    except Exception:
                        sample_base_freq = 220.0
                st.caption(f"Detected/used sample base pitch: {sample_base_freq:.1f} Hz")
        else:
            notes = note_options()
            note_label = st.selectbox(
                "Base note for the synthesized tone", [n for n, _ in notes],
                index=[n for n, _ in notes].index("A3") if "A3" in [n for n, _ in notes] else 0,
            )
            base_note_freq = dict(notes)[note_label]

        if st.button("Apply instrument conversion"):
            with st.spinner("Detecting onsets/pitch and resynthesizing..."):
                y, sr = librosa.load(orig_path, sr=None, mono=True)
                new_mono = resynthesize_segment(
                    y, sr, start_s, end_s, detect_mode,
                    source="sample" if source_choice == "Upload a sample" and sample_y is not None else "synth",
                    sample_y=sample_y, sample_sr=sample_sr, sample_base_freq=sample_base_freq,
                    base_note_freq=base_note_freq, max_note_ms=max_note_ms,
                )
                combined = splice_segment(
                    st.session_state.stems[stem_to_edit], new_mono, sr, start_s, end_s
                )
                work_dir = st.session_state.work_dir
                out_path = os.path.join(work_dir, f"{stem_to_edit}_modified.wav")
                combined.export(out_path, format="wav")
                st.session_state.stems[stem_to_edit] = out_path
                st.success("Stem converted.")
                st.rerun()

    if st.session_state.stems[stem_to_edit] != orig_path:
        if st.button("Reset this stem to original separated audio"):
            st.session_state.stems[stem_to_edit] = orig_path
            st.rerun()

    # -----------------------------------------------------------------
    # 5. Render final mix
    # -----------------------------------------------------------------
    st.header("5. Render final mix")
    if st.button("Mix all stems into final track"):
        with st.spinner("Mixing..."):
            final_mix = None
            for stem, path in st.session_state.stems.items():
                seg = AudioSegment.from_file(path)
                final_mix = seg if final_mix is None else final_mix.overlay(seg)
            out_path = os.path.join(st.session_state.work_dir, "final_mix.wav")
            final_mix.export(out_path, format="wav")

            st.success("Done! Here is your remixed track:")
            st.audio(out_path)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download final mix", f, file_name="final_mix.wav", mime="audio/wav")
