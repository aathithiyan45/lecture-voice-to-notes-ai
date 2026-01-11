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
    # For quiz/flashcard generation
    return pipeline("text2text-generation", model="google/flan-t5-base")

summarizer = load_summarizer()
generator = load_generator()


# ---------------------------
# AUDIO CONVERSION
# ---------------------------
def convert_to_wav(input_path: str) -> str:
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
# SUMMARIZATION
# ---------------------------
def summarize_text(text: str) -> str:
    words = text.split()
    if len(words) < 40:
        return text

    chunk_size = 380
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    summaries = []
    for chunk in chunks:
        out = summarizer(chunk, max_length=130, min_length=50, do_sample=False)
        summaries.append(out[0]["summary_text"])

    return "\n\n".join(summaries)


# ---------------------------
# QUIZ + FLASHCARDS GENERATION
# ---------------------------
def generate_mcqs(notes: str, n=5) -> str:
    prompt = f"""
Generate {n} multiple-choice questions (MCQs) from the below notes.
Format exactly like this:

Q1) Question?
A) option
B) option
C) option
D) option
Answer: A

NOTES:
{notes}
"""
    out = generator(prompt, max_length=512, do_sample=False)
    return out[0]["generated_text"]

def generate_true_false(notes: str, n=5) -> str:
    prompt = f"""
Generate {n} True/False questions from the notes.
Format:

1) Statement - True/False: __

NOTES:
{notes}
"""
    out = generator(prompt, max_length=512, do_sample=False)
    return out[0]["generated_text"]

def generate_one_mark(notes: str, n=5) -> str:
    prompt = f"""
Generate {n} one-mark questions from the notes (short answer questions).
Format:

1) Question?
Answer: ...

NOTES:
{notes}
"""
    out = generator(prompt, max_length=512, do_sample=False)
    return out[0]["generated_text"]

def generate_flashcards(notes: str, n=5) -> str:
    prompt = f"""
Create {n} flashcards from the notes.
Format:

Card 1:
Front: ...
Back: ...

NOTES:
{notes}
"""
    out = generator(prompt, max_length=512, do_sample=False)
    return out[0]["generated_text"]


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

        # Quiz + Flashcards Section
        st.subheader("🧠 Quiz & Flashcards (AI Generated)")

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
            st.text_area("✅ MCQs", mcqs, height=300)

        if tf_btn:
            st.info("⏳ Generating True/False...")
            tf = generate_true_false(notes, n=5)
            st.text_area("✅ True/False", tf, height=250)

        if one_btn:
            st.info("⏳ Generating One-Mark Questions...")
            one = generate_one_mark(notes, n=5)
            st.text_area("✅ One-Mark Questions", one, height=250)

        if flash_btn:
            st.info("⏳ Generating Flashcards...")
            flash = generate_flashcards(notes, n=5)
            st.text_area("✅ Flashcards", flash, height=300)

        st.markdown("---")

        # Download outputs
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
