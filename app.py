import os
import tempfile
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from transformers import pipeline


# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(
    page_title="Lecture Voice-to-Notes Generator",
    page_icon="🎙️",
    layout="centered"
)


# ---------------------------
# LOAD AI MODELS (cached)
# ---------------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_generator():
    # Smaller model = faster on M1, less download
    return pipeline("text2text-generation", model="google/flan-t5-small")

summarizer = load_summarizer()
generator = load_generator()


# ---------------------------
# AUDIO CONVERSION
# ---------------------------
def convert_to_wav(input_path: str) -> str:
    """
    Converts any audio format supported by ffmpeg/pydub to
    16kHz mono PCM wav for SpeechRecognition.
    """
    sound = AudioSegment.from_file(input_path)
    sound = sound.set_channels(1).set_frame_rate(16000)

    output_path = os.path.join(os.path.dirname(input_path), "converted.wav")
    sound.export(output_path, format="wav")
    return output_path


# ---------------------------
# SPEECH TO TEXT
# ---------------------------
def speech_to_text(audio_path: str) -> str:
    recognizer = sr.Recognizer()
    wav_path = convert_to_wav(audio_path)

    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "❌ Sorry, could not understand the audio."
    except sr.RequestError:
        return "❌ Speech-to-text request failed. Check your internet connection."


# ---------------------------
# SUMMARIZATION (NOTES)
# ---------------------------
def summarize_text(text: str) -> str:
    words = text.split()
    if len(words) < 40:
        return text

    chunk_size = 380
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    summaries = []
    for chunk in chunks:
        # dynamic lengths to avoid warnings
        chunk_words = len(chunk.split())
        max_len = min(120, max(40, chunk_words // 2))

        out = summarizer(
            chunk,
            max_length=max_len,
            min_length=30,
            do_sample=False
        )
        summaries.append(out[0]["summary_text"])

    return "\n\n".join(summaries)


# ---------------------------
# QUIZ + FLASHCARDS GENERATION
# ---------------------------
def generate_mcqs(notes: str, n=5) -> str:
    prompt = f"""
You are an exam question generator.

Generate {n} MCQ questions from the notes.
Each question MUST follow this exact format:

Q1. <question>
A. <option>
B. <option>
C. <option>
D. <option>
Answer: <A/B/C/D>

NOTES:
{notes}
"""
    out = generator(prompt, max_new_tokens=256, do_sample=False)
    text = out[0]["generated_text"]

    # formatting improvement
    text = text.replace("Q", "\nQ").strip()
    return text


def generate_true_false(notes: str, n=5) -> str:
    prompt = f"""
Generate {n} True/False questions from the notes.
Format exactly:

1. <statement>
Answer: True/False

NOTES:
{notes}
"""
    out = generator(prompt, max_new_tokens=256, do_sample=False)
    return out[0]["generated_text"].strip()


def generate_one_mark(notes: str, n=5) -> str:
    prompt = f"""
Generate {n} one-mark (short answer) questions from the notes.
Format:

1. Question?
Answer: ...

NOTES:
{notes}
"""
    out = generator(prompt, max_new_tokens=256, do_sample=False)
    return out[0]["generated_text"].strip()


def generate_flashcards(notes: str, n=5) -> str:
    prompt = f"""
Create {n} flashcards from the notes.
Format exactly:

Card 1:
Front: ...
Back: ...

NOTES:
{notes}
"""
    out = generator(prompt, max_new_tokens=256, do_sample=False)
    return out[0]["generated_text"].strip()


# ---------------------------
# UI
# ---------------------------
st.title("🎙️ Lecture Voice-to-Notes Generator")
st.write("Upload lecture audio and get **Transcript + Summarized Notes + Quiz/Flashcards** using AI (Prototype-level system).")

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Lecture Audio (mp3 / m4a / wav)",
    type=["mp3", "m4a", "wav"]
)

if uploaded_file is not None:
    st.success("✅ Audio uploaded successfully!")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_audio_path = os.path.join(tmpdir, uploaded_file.name)

        with open(input_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Transcript
        st.info("⏳ Converting audio to text...")
        transcript = speech_to_text(input_audio_path)

        st.subheader("📌 Transcript")
        st.write(transcript)

        st.markdown("---")

        # Notes
        st.info("⏳ Generating summarized notes...")
        notes = summarize_text(transcript)

        st.subheader("📝 Summarized Notes")
        st.write(notes)

        st.markdown("---")

        # Quiz + Flashcards
        st.subheader("🧠 Quiz & Flashcards (AI Generated)")
        st.caption("Choose a feature to generate practice materials for revision.")

        col1, col2 = st.columns(2)
        with col1:
            mcq_btn = st.button("Generate MCQs")
            tf_btn = st.button("Generate True/False")
        with col2:
            one_btn = st.button("Generate One-Mark Qs")
            flash_btn = st.button("Generate Flashcards")

        if mcq_btn:
            st.info("⏳ Generating MCQs...")
            mcqs = generate_mcqs(notes, n=5)
            st.subheader("✅ MCQs")
            st.code(mcqs)

        if tf_btn:
            st.info("⏳ Generating True/False...")
            tf = generate_true_false(notes, n=5)
            st.subheader("✅ True/False")
            st.code(tf)

        if one_btn:
            st.info("⏳ Generating One-Mark Questions...")
            one = generate_one_mark(notes, n=5)
            st.subheader("✅ One-Mark Questions")
            st.code(one)

        if flash_btn:
            st.info("⏳ Generating Flashcards...")
            flash = generate_flashcards(notes, n=5)
            st.subheader("✅ Flashcards")
            st.code(flash)

        st.markdown("---")

        # Downloads
        st.subheader("⬇️ Download Outputs")

        st.download_button(
            label="Download Transcript (.txt)",
            data=transcript,
            file_name="transcript.txt",
            mime="text/plain"
        )

        st.download_button(
            label="Download Notes (.txt)",
            data=notes,
            file_name="notes.txt",
            mime="text/plain"
        )

st.markdown("---")
st.caption("⚡ Developed by Aathithiyan P | IDM / AICTE Internship Project")
