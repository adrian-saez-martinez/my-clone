import requests
import streamlit as st
import os
import glob
from data.data_utils import process_and_upload_text

# Hugging Face Whisper API
WHISPER_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
HF_TOKEN = st.secrets["HF_TOKEN"]
WHISPER_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_whisper(audio_bytes):
    """Send audio data to Hugging Face Whisper API for transcription."""
    response = requests.post(WHISPER_API_URL, headers=WHISPER_HEADERS, data=audio_bytes)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def bulk_process_audios(audio_folder, data_folder, db_root):
    """
    Bulk process audio files, transcribe them, and upload to Chroma databases.

    Args:
        audio_folder (str): Path to the folder containing audio files.
        data_folder (str): Path to the folder where transcriptions will be saved.
        db_root (str): Path to the root folder containing Chroma databases.

    Returns:
        None
    """
    # Get all audio files in the folder
    audio_files = glob.glob(os.path.join(audio_folder, "*.mp3")) + glob.glob(os.path.join(audio_folder, "*.wav"))

    for audio_file in audio_files:
        # Extract topic from the file name
        file_name = os.path.basename(audio_file)
        topic, _ = file_name.split("_", 1)  # Get topic before the first "_"

        # Transcribe audio
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()

        transcription_response = query_whisper(audio_bytes)

        if "text" in transcription_response:
            transcription = transcription_response["text"]
            print(f"Transcription for {file_name}:\n{transcription}")

            # Save transcription to file
            save_path = os.path.join(data_folder, f"{topic}_{file_name}.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f"Topic: {topic}\n\n")
                f.write(transcription)
            print(f"Transcription saved to {save_path}")

            # Determine database path
            db_path = os.path.join(db_root, f"{topic}_db")

            # Upload transcription to Chroma
            process_and_upload_text(save_path, db_path)
            print(f"Transcription uploaded to {topic} database at {db_path}")
        else:
            print(f"Failed to transcribe {file_name}: No 'text' in response.")


audio_folder = "data/audios/"  # Folder containing audio files
data_folder = "data/texts/"  # Folder to save transcriptions
db_root = "./chroma_databases/"  # Root folder for Chroma databases

# Run bulk processing
bulk_process_audios(audio_folder, data_folder, db_root)
