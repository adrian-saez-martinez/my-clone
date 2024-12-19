import streamlit as st
import datetime
import requests
from data.data_utils import process_and_upload_text

# Hugging Face Whisper API configuration
WHISPER_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
HF_TOKEN = st.secrets["HF_TOKEN"]
WHISPER_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_whisper(audio_bytes):
    """Send audio data to Hugging Face whisper API for transcription."""
    response = requests.post(WHISPER_API_URL, headers=WHISPER_HEADERS, data=audio_bytes)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return None

def admin_page():
    """Admin interface for adding data to databases."""
    st.title("Admin Panel: Add Data to Databases")

    # Topic Selector
    topic = st.selectbox("Choose a topic:", ["resume", "thoughts", "personal"])
    fecha = datetime.date.today()

    # Section 1: Upload or Record Audio
    st.subheader("Record or Upload Audio")
    audio_value = st.file_uploader("Upload audio (mp3, wav):", type=["mp3", "wav"])

    if audio_value:
        st.audio(audio_value)

        # Step 1: Transcription
        transcription_response = query_whisper(audio_value.read())

        if transcription_response and "text" in transcription_response:
            transcription = transcription_response["text"]
            st.subheader("Audio Transcription:")
            st.write(transcription)

            # Save transcription to file
            save_path = f"data/{topic}_{fecha}.txt"
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f"Topic: {topic}\nDate: {fecha.strftime('%d/%m/%Y')}\n\n")
                f.write(transcription)
            st.success(f"Transcription saved to {save_path}")

    # Section 2: Upload Text File
    st.subheader("Upload Text File")
    text_file = st.file_uploader("Upload text file (.txt):", type=["txt"])

    if text_file:
        # Save text file
        save_path = f"data/{text_file.name}"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text_file.read().decode("utf-8"))
        st.success(f"Text file saved to {save_path}")

        # Optional Upload
        st.subheader("Upload to Database")
        if st.button("Upload Text File"):
            db_path = f"./chroma_databases/{topic}_db/"
            process_and_upload_text(save_path, db_path)
            st.success(f"Text file uploaded to {topic} chroma database successfully!")
