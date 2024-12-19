import os
from utils.data_utils import process_and_upload_text

def bulk_update_chroma(data_folder, db_root, topics):
    """
    Bulk update Chroma databases with text files collected for each topic.

    Args:
        data_folder (str): Path to the folder containing text files.
        db_root (str): Path to the root folder for Chroma databases.
        topics (list): List of topics (e.g., ["resume", "thoughts", "personal", "contact"]).

    Returns:
        None
    """
    for topic in topics:
        print(f"Processing topic: {topic}")

        # Define paths
        topic_folder = os.path.join(data_folder, topic)
        db_path = os.path.join(db_root, f"{topic}_db")

        # Ensure the topic folder exists
        if not os.path.exists(topic_folder):
            print(f"Folder for topic '{topic}' not found in {data_folder}. Skipping...")
            continue

        # Process all .txt files in the topic folder
        for file_name in os.listdir(topic_folder):
            if file_name.endswith(".txt"):
                file_path = os.path.join(topic_folder, file_name)
                print(f"Processing file: {file_path}")

                # Upload the text file to the corresponding Chroma database
                process_and_upload_text(file_path, db_path)
                print(f"Uploaded {file_name} to {db_path}")

    print("Bulk update completed.")


# Define folders and topics
data_folder = "data/texts/"  # Folder containing topic subfolders
db_root = "./chroma_databases/"  # Root folder for Chroma databases
topics = ["resume", "thoughts", "personal", "contact"]  # List of topics

# Run bulk update
bulk_update_chroma(data_folder, db_root, topics)

