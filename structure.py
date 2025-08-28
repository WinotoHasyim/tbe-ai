# ==============================================================================
# --- 1. SETUP AND AUTHENTICATION FOR GOOGLE COLAB ---
# ==============================================================================

# Install necessary libraries
!pip install exifread Pillow webdavclient3 tqdm

import os
import logging
import requests
import exifread
import time
from io import BytesIO
from datetime import datetime
from webdav3 import client as wc
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# --- Third-party libraries ---
from google.colab import auth
from google.auth import default
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from PIL import Image
from tqdm.notebook import tqdm

# --- Colab Specific Setup ---
# Authenticate the user. This is required for the API to work.
auth.authenticate_user()

# Access secrets
from google.colab import userdata

# ==============================================================================
# --- 2. CONFIGURATION ---
# ==============================================================================
# --- Google Drive Settings ---
SOURCE_ROOT_FOLDER_ID = userdata.get("SOURCE_ROOT_FOLDER_ID")
NANDA_FOLDER_ID = userdata.get("NANDA_FOLDER_ID") # Folder with pre-compressed JPEGs
# --- EDITED: Add a folder ID for storing the log file ---
LOG_FOLDER_ID = userdata.get("LOG_FOLDER_ID")

# --- Nextcloud Settings (Destination) ---
NEXTCLOUD_HOSTNAME = userdata.get("NEXTCLOUD_HOSTNAME")
NEXTCLOUD_USERNAME = userdata.get("NEXTCLOUD_USERNAME")
NEXTCLOUD_PASSWORD = userdata.get("NEXTCLOUD_PASSWORD")
NEXTCLOUD_ROOT_PATH = userdata.get("NEXTCLOUD_ROOT_PATH")
FAILED_API_FOLDER_NAME = "FAILED"

# --- TBE API Settings ---
TBE_API_URL = userdata.get("TBE_API_URL")
TBE_API_KEY = userdata.get("TBE_API_KEY")

# --- State Management ---
STATE_FILE = "processed_files.log"
MAX_RUNTIME_SECONDS = 11 * 3600 + 45 * 60  # 5 hours and 45 minutes

# --- Processing Logic Settings ---
MONITORING_KEYWORDS = ["tikus", "tanaman", "ngengat", "ulat", "penggerek"]

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().setLevel(logging.INFO)

# ==============================================================================
# --- 3. HELPER FUNCTIONS ---
# ==============================================================================
def get_decimal_from_dms(dms, ref):
    """Converts GPS coordinates from DMS (degrees, minutes, seconds) to decimal."""
    degrees = dms.values[0].num / dms.values[0].den
    minutes = dms.values[1].num / dms.values[1].den / 60.0
    seconds = dms.values[2].num / dms.values[2].den / 3600.0
    decimal = degrees + minutes + seconds
    if ref.values in ['S', 'W']:
        decimal = -decimal
    return decimal

def get_lat_lon_and_datetime(dng_byte_stream):
    """Extracts GPS Lat/Lon and creation datetime from a DNG byte stream."""
    lat, lon, dt = None, None, None
    try:
        dng_byte_stream.seek(0)
        tags = exifread.process_file(dng_byte_stream, details=False)
        if 'EXIF DateTimeOriginal' in tags:
            dt_str = str(tags['EXIF DateTimeOriginal'])
            dt = datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
        lat_dms = tags.get('GPS GPSLatitude')
        lat_ref = tags.get('GPS GPSLatitudeRef')
        lon_dms = tags.get('GPS GPSLongitude')
        lon_ref = tags.get('GPS GPSLongitudeRef')
        if all([lat_dms, lat_ref, lon_dms, lon_ref]):
            lat = get_decimal_from_dms(lat_dms, lat_ref)
            lon = get_decimal_from_dms(lon_dms, lon_ref)
    except Exception:
        return None, None, None
    return lat, lon, dt

def list_files_with_retry(service, query, page_token):
    """
    Executes the files().list() API call with a retry mechanism for transient errors.
    """
    max_retries = 5
    delay = 1  # Initial delay in seconds
    for attempt in range(max_retries):
        try:
            return service.files().list(
                q=query,
                pageSize=1000,
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token
            ).execute()
        except HttpError as error:
            if error.resp.status in [500, 502, 503, 504]:
                logging.warning(f"API call failed with transient error {error.resp.status}. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise
    logging.error(f"API call failed after {max_retries} retries.")
    return None # Return None if all retries fail


def map_files_in_folder(service, folder_id):
    """
    Scans a Google Drive folder recursively and creates a map of
    basename -> list of {'id': file_id, 'path': file_path} to handle duplicates.
    """
    logging.info(f"Mapping all files in Google Drive folder {folder_id}...")
    file_map = defaultdict(list)

    def _recursive_map(folder_id_to_scan, current_path=""):
        if not current_path:
            try:
                root_folder = service.files().get(fileId=folder_id_to_scan, fields='name').execute()
                current_path = root_folder.get('name', 'ROOT')
            except HttpError:
                current_path = "ROOT"

        query = f"'{folder_id_to_scan}' in parents and trashed = false"
        page_token = None
        while True:
            try:
                results = list_files_with_retry(service, query, page_token)
                if not results:
                    break
                items = results.get('files', [])
                for item in items:
                    item_path = os.path.join(current_path, item['name']).replace("\\", "/")
                    if item['mimeType'] == 'application/vnd.google-apps.folder':
                        _recursive_map(item['id'], item_path)
                    else:
                        base_name = os.path.splitext(item['name'])[0]
                        file_map[base_name].append({'id': item['id'], 'path': os.path.splitext(item_path)[0]})
                page_token = results.get('nextPageToken')
                if not page_token: break
            except HttpError as error:
                logging.error(f"Could not scan folder ID {folder_id_to_scan}. Error: {error}")
                break

    _recursive_map(folder_id)
    total_paths = sum(len(v) for v in file_map.values())
    logging.info(f"Finished mapping. Found {total_paths} different file paths.")
    return file_map

# --- Nextcloud Write Helpers ---
def get_nextcloud_client():
    """Initializes and returns a WebDAV client for Nextcloud."""
    options = {
        'webdav_hostname': NEXTCLOUD_HOSTNAME,
        'webdav_login': NEXTCLOUD_USERNAME,
        'webdav_password': NEXTCLOUD_PASSWORD
    }
    try:
        client = wc.Client(options)
        client.verify = True
        return client
    except Exception:
        return None

def ensure_nextcloud_folder(client, remote_path):
    """Checks if a folder exists on Nextcloud and creates it if not."""
    try:
        if not client.check(remote_path):
            client.mkdir(remote_path)
        return True
    except Exception:
        return False

def upload_to_nextcloud(gdrive_service, source_file_id, remote_path):
    """Downloads a file from Google Drive into memory and uploads it to Nextcloud using requests."""
    try:
        request = gdrive_service.files().get_media(fileId=source_file_id)
        file_bytes = BytesIO(request.execute())
        file_bytes.seek(0)
        full_url = f"{NEXTCLOUD_HOSTNAME}/{remote_path.lstrip('/')}"
        response = requests.put(
            full_url,
            data=file_bytes,
            auth=(NEXTCLOUD_USERNAME, NEXTCLOUD_PASSWORD),
            headers={'Content-Type': 'application/octet-stream'}
        )
        response.raise_for_status()
        return True
    except Exception:
        return False

# ==============================================================================
# --- 4. CORE PROCESSING LOGIC ---
# ==============================================================================
def process_dng_file(gdrive_service, nextcloud_client, file_info, original_path, compressed_file_map):
    """
    Main ETL logic for a single DNG file. Returns a status tuple.
    The actual logging to a file is now handled by the main loop.
    """
    file_id = file_info['id']
    file_name = file_info['name']
    base_name = os.path.splitext(file_name)[0]

    candidates = compressed_file_map.get(base_name)
    if not candidates:
        return ('unmatched', original_path)

    compressed_file_id = None
    if len(candidates) == 1:
        compressed_file_id = candidates[0]['id']
    else:
        dng_parent_path = os.path.dirname(original_path)
        for candidate in candidates:
            jpeg_parent_path = os.path.dirname(candidate['path'])
            if dng_parent_path == jpeg_parent_path:
                compressed_file_id = candidate['id']
                break

    if not compressed_file_id:
        return ('unmatched_duplicate', original_path)

    try:
        request = gdrive_service.files().get_media(fileId=file_id)
        dng_bytes = BytesIO(request.execute())
        lat, lon, dt = get_lat_lon_and_datetime(dng_bytes)
        if not all([lat, lon, dt]):
            return ('no_metadata', file_name)
    except Exception:
        return ('download_error', file_name)

    api_success = False
    api_data = {}
    api_failure_reason = "Unknown API error"
    try:
        if not TBE_API_KEY: raise ValueError("TBE_API_KEY secret is missing.")
        params = {'lat': lat, 'lng': lon}
        headers = {'Key': TBE_API_KEY}
        response = requests.get(TBE_API_URL, params=params, headers=headers, timeout=15)
        if response.status_code == 200 and response.json().get('status') is True:
            api_data = response.json().get('data', {})
            if api_data.get('land_id'):
                api_success = True
            else:
                api_failure_reason = "API returned success but no land_id"
        else:
            api_failure_reason = f"API returned status {response.status_code}: {response.text}"
    except (requests.exceptions.RequestException, ValueError) as e:
        api_failure_reason = f"API request exception: {e}"

    new_filename = f"{original_path.replace('/', '_')}.jpeg"

    if api_success:
        land_id = api_data.get('land_id')
        day_val = api_data.get('land_hst', '0') if api_data.get('land_hst', '0') != '0' else api_data.get('land_hss', '0')
        main_folder_path = f"{NEXTCLOUD_ROOT_PATH}/{land_id}-{day_val}"
        if ensure_nextcloud_folder(nextcloud_client, main_folder_path):
            for keyword in MONITORING_KEYWORDS:
                ensure_nextcloud_folder(nextcloud_client, f"{main_folder_path}/monitoring {keyword}")
        target_subfolder_path = f"{main_folder_path}/monitoring tanaman"
        for keyword in MONITORING_KEYWORDS:
            if keyword in original_path.lower():
                target_subfolder_path = f"{main_folder_path}/monitoring {keyword}"
                break
        if upload_to_nextcloud(gdrive_service, compressed_file_id, f"{target_subfolder_path}/{new_filename}"):
            return ('success', file_id)
        else:
            return ('upload_fail', file_name)
    else:
        logging.warning(f"Uploading '{file_name}' to failure folder. Reason: {api_failure_reason}")
        failure_path = f"{NEXTCLOUD_ROOT_PATH}/{FAILED_API_FOLDER_NAME}"
        if ensure_nextcloud_folder(nextcloud_client, failure_path):
            if upload_to_nextcloud(gdrive_service, compressed_file_id, f"{failure_path}/{new_filename}"):
                return ('api_fail_upload_success', file_id)
        return ('upload_fail', file_name)


def traverse_drive(service, folder_id, current_path=""):
    """Recursively traverses Google Drive, yielding file info and path."""
    if not current_path:
        try:
            root_folder = service.files().get(fileId=folder_id, fields='name').execute()
            current_path = root_folder.get('name', 'ROOT')
        except HttpError:
            current_path = "ROOT"

    query = f"'{folder_id}' in parents and trashed = false"
    page_token = None
    while True:
        try:
            results = list_files_with_retry(service, query, page_token)
            if not results:
                break
            items = results.get('files', [])
            for item in items:
                item_path = os.path.join(current_path, item['name']).replace("\\", "/")
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    yield from traverse_drive(service, item['id'], item_path)
                elif item['name'].lower().endswith('.dng'):
                    yield item, os.path.splitext(item_path)[0]
            page_token = results.get('nextPageToken')
            if not page_token:
                break
        except HttpError as error:
            logging.error(f"Could not traverse folder ID {folder_id}. Error: {error}")
            break

# ==============================================================================
# --- 5. MAIN EXECUTION ---
# ==============================================================================
def load_processed_files():
    """Loads the set of already processed file IDs from the local state file."""
    if not os.path.exists(STATE_FILE):
        return set()
    with open(STATE_FILE, 'r') as f:
        return {line.strip() for line in f if line.strip()}

def find_log_file_on_drive(service, folder_id, filename):
    """Searches for the log file on Google Drive and returns its ID if found."""
    query = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
    try:
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        for file in response.get('files', []):
            return file.get('id')
    except HttpError as error:
        logging.error(f"Could not search for log file: {error}")
    return None

def download_log_from_drive(service, file_id, local_path):
    """Downloads the log file from Google Drive to the local Colab environment."""
    try:
        request = service.files().get_media(fileId=file_id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        with open(local_path, 'wb') as f:
            f.write(fh.getvalue())
        logging.info(f"Successfully downloaded log file from Drive to '{local_path}'.")
        return True
    except HttpError as error:
        logging.error(f"Could not download log file: {error}")
        return False

def update_log_on_drive(service, file_id, folder_id, local_path):
    """Creates or updates the log file on Google Drive."""
    media = MediaFileUpload(local_path, mimetype='text/plain')
    try:
        if file_id:
            # File exists, update it
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            # File doesn't exist, create it
            file_metadata = {'name': os.path.basename(local_path), 'parents': [folder_id]}
            file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            return file.get('id') # Return the new file ID
    except HttpError as error:
        logging.error(f"Could not upload log file to Drive: {error}")
    return file_id # Return old ID on failure, or new ID on initial success


def main():
    """Main function to run the script with persistent logging to Google Drive."""
    start_time = time.monotonic()

    creds, _ = default()
    gdrive_service = build('drive', 'v3', credentials=creds)
    nextcloud_client = get_nextcloud_client()

    if not nextcloud_client:
        logging.critical("Could not connect to Nextcloud. Aborting.")
        return

    try:
        # --- EDITED: Manage log file on Google Drive ---
        log_file_id = find_log_file_on_drive(gdrive_service, LOG_FOLDER_ID, STATE_FILE)
        if log_file_id:
            download_log_from_drive(gdrive_service, log_file_id, STATE_FILE)
        else:
            logging.info(f"No existing log file found on Google Drive. A new one will be created.")

        processed_ids = load_processed_files()
        logging.info(f"Loaded {len(processed_ids)} previously processed file IDs.")

        compressed_file_map = map_files_in_folder(gdrive_service, NANDA_FOLDER_ID)
        logging.info(f"Starting traversal from GDrive folder: '{SOURCE_ROOT_FOLDER_ID}'")
        all_tasks = list(traverse_drive(gdrive_service, SOURCE_ROOT_FOLDER_ID))
        tasks_to_run = [task for task in all_tasks if task[0]['id'] not in processed_ids]
        logging.info(f"Found {len(all_tasks)} total DNG files. {len(tasks_to_run)} files remaining to process.")

        unmatched_dngs = []

        for task in tqdm(tasks_to_run, desc="Processing Files"):
            if time.monotonic() - start_time > MAX_RUNTIME_SECONDS:
                logging.warning(f"Runtime limit reached. Stopping script.")
                break

            file_info, original_path = task
            status, detail = process_dng_file(gdrive_service, nextcloud_client, file_info, original_path, compressed_file_map)

            if status in ['success', 'api_fail_upload_success']:
                # --- EDITED: Robust local save and update to Google Drive ---
                with open(STATE_FILE, 'a') as f:
                    f.write(f"{detail}\n")
                    f.flush()
                    os.fsync(f.fileno())
                # Now, upload the updated log file to Google Drive
                new_log_id = update_log_on_drive(gdrive_service, log_file_id, LOG_FOLDER_ID, STATE_FILE)
                if new_log_id:
                    log_file_id = new_log_id # Update file ID in case it was just created

            elif status.startswith('unmatched'):
                unmatched_dngs.append(detail)

        if unmatched_dngs:
            with open("unmatched_dngs.txt", "w") as f:
                f.write(f"Found {len(unmatched_dngs)} DNG files with no matching compressed JPEG:\n")
                for path in unmatched_dngs:
                    f.write(f" - {path}.dng\n")
            logging.warning(f"Summary of unmatched files written to unmatched_dngs.txt")

        logging.info("✅ Processing complete.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in main: {e}", exc_info=True)

# Run the main function
main()
