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
# LOAD SUMMARIZER (cached)
# ---------------------------
@st.cache_resource
def load_summarizer():
    # Reliable summarization model
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()


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
    """
    Summarize long lecture transcript into study notes.
    Handles long input by splitting into chunks.
    """
    words = text.split()

    # If transcript is too short, summarization not needed
    if len(words) < 40:
        return text

    # Chunking to avoid model max token limit
    chunk_size = 380  # safe chunk size (words)
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    summaries = []
    for chunk in chunks:
        out = summarizer(chunk, max_length=130, min_length=50, do_sample=False)
        summaries.append(out[0]["summary_text"])

    final_notes = "\n\n".join(summaries)
    return final_notes


# ---------------------------
# UI
# ---------------------------
st.title("🎙️ Lecture Voice-to-Notes Generator")
st.write("Upload lecture audio and get **Transcript + Summarized Notes** using AI (Prototype-level system).")

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Lecture Audio (mp3 / m4a / wav)",
    type=["mp3", "m4a", "wav"]
)

if uploaded_file is not None:
    st.success("✅ Audio uploaded successfully!")

    # Save uploaded file to a temp location
    with tempfile.TemporaryDirectory() as tmpdir:
        input_audio_path = os.path.join(tmpdir, uploaded_file.name)

        with open(input_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Convert + Transcript
        st.info("⏳ Converting audio to text...")
        transcript = speech_to_text(input_audio_path)

        st.subheader("📌 Transcript")
        st.write(transcript)

        st.markdown("---")

        # Summarize
        st.info("⏳ Generating summarized notes...")
        notes = summarize_text(transcript)

        st.subheader("📝 Summarized Notes")
        st.write(notes)

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
