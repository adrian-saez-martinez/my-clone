import os
from data_utils import process_and_upload_text

def bulk_update_allinfo_chroma(data_folder, db_path):
    """
    Bulk update a single Chroma database with all text files from multiple topic folders.

    Args:
        data_folder (str): Path to the folder containing text files organized in topic subfolders.
        db_path (str): Path to the Chroma database for all information.

    Returns:
        None
    """
    print(f"Updating Chroma database at: {db_path}")

    # Iterate through topic subfolders
    for topic in os.listdir(data_folder):
        topic_folder = os.path.join(data_folder, topic)

        # Ensure the topic folder exists and is a directory
        if not os.path.isdir(topic_folder):
            print(f"Skipping non-directory item: {topic}")
            continue

        print(f"Processing topic folder: {topic}")

        # Process all .txt files in the topic folder
        for file_name in os.listdir(topic_folder):
            if file_name.endswith(".txt"):
                file_path = os.path.join(topic_folder, file_name)
                print(f"Processing file: {file_path}")

                # Upload the text file to the single Chroma database
                process_and_upload_text(file_path, db_path)
                print(f"Uploaded {file_name} from topic '{topic}' to {db_path}")

    print("Bulk update to single Chroma database completed.")


# Define folder and database path
data_folder = "data/texts/"  # Folder containing topic subfolders
db_path = "./chroma_databases/allinfo_db"  # Path to the single Chroma database

# Run bulk update
bulk_update_allinfo_chroma(data_folder, db_path)
