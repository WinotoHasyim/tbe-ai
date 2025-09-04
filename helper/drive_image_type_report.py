# ==============================================================================
# --- 1. SETUP ---
# ==============================================================================

!pip install google-api-python-client

import os
import logging
from collections import Counter
from google.colab import auth
from google.auth import default
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Authenticate Colab with Google
auth.authenticate_user()
creds, _ = default()
drive_service = build('drive', 'v3', credentials=creds)

# Access secrets
from google.colab import userdata

# --- Google Drive Source Folder ---
SOURCE_ROOT_FOLDER_ID = userdata.get("SOURCE_ROOT_FOLDER_ID")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def list_files_with_retry(service, query, page_token=None):
    """Helper with retry for Drive API list."""
    try:
        return service.files().list(
            q=query,
            pageSize=1000,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token
        ).execute()
    except HttpError as e:
        logging.error(f"Drive API error: {e}")
        return None

def traverse_drive(service, folder_id, ext_counter):
    """Recursively traverse a Drive folder and count file extensions."""
    query = f"'{folder_id}' in parents and trashed=false"
    page_token = None

    while True:
        results = list_files_with_retry(service, query, page_token)
        if not results:
            break

        items = results.get("files", [])
        for item in items:
            if item["mimeType"] == "application/vnd.google-apps.folder":
                traverse_drive(service, item["id"], ext_counter)
            else:
                _, ext = os.path.splitext(item["name"].lower())
                if ext:
                    ext_counter[ext] += 1
                else:
                    ext_counter["(no extension)"] += 1

        page_token = results.get("nextPageToken")
        if not page_token:
            break

# ==============================================================================
# --- 3. MAIN ---
# ==============================================================================

def main():
    logging.info(f"Scanning Drive folder {SOURCE_ROOT_FOLDER_ID} for image types...")
    ext_counter = Counter()
    traverse_drive(drive_service, SOURCE_ROOT_FOLDER_ID, ext_counter)

    print("\n--- Image Type Report ---")
    for ext, count in ext_counter.most_common():
        print(f"{ext}: {count} files")
    print("-------------------------")

main()
