# ==============================================================================
# --- 1. SETUP AND AUTHENTICATION FOR GOOGLE COLAB ---
# ==============================================================================

# Install necessary libraries
# Added rawpy and opencv-python-headless for image compression
!pip install exifread Pillow webdavclient3 tqdm rawpy opencv-python-headless numpy

import os
import logging
import requests
import exifread
import time
import rawpy
import cv2
import numpy as np
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
# The source folder containing the DNG files
SOURCE_ROOT_FOLDER_ID = userdata.get("SOURCE_ROOT_FOLDER_ID")
# The folder where the processed.log file is stored
LOG_FOLDER_ID = userdata.get("LOG_FOLDER_ID")
# NOTE: NANDA_FOLDER_ID is no longer needed as we compress on-the-fly.

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
# Set a max runtime to avoid Colab timeouts (e.g., 11 hours, 45 minutes)
MAX_RUNTIME_SECONDS = 11 * 3600 + 45 * 60

# --- Processing Logic Settings ---
MONITORING_KEYWORDS = ["tikus", "tanaman", "ngengat", "ulat", "penggerek"]
# Target JPEG size in MB
MIN_JPEG_SIZE_MB = 1.0
MAX_JPEG_SIZE_MB = 5.0


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().setLevel(logging.INFO)

# ==============================================================================
# --- 3. HELPER FUNCTIONS ---
# ==============================================================================

# --- GPS and Metadata Helpers (from structure.py) ---
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
    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")
        return None, None, None
    return lat, lon, dt

# --- Google Drive API Helpers (from structure.py) ---
def list_files_with_retry(service, query, page_token):
    """Executes the files().list() API call with an exponential backoff retry mechanism."""
    max_retries = 5
    delay = 1
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
                logging.warning(f"API call failed with transient error {error.resp.status}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise
    logging.error(f"API call failed after {max_retries} retries.")
    return None

# --- Nextcloud WebDAV Helpers (from structure.py) ---
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
    except Exception as e:
        logging.error(f"Failed to initialize Nextcloud client: {e}")
        return None

def ensure_nextcloud_folder(client, remote_path):
    """Checks if a folder exists on Nextcloud and creates it if not."""
    try:
        if not client.check(remote_path):
            client.mkdir(remote_path)
        return True
    except Exception as e:
        logging.error(f"Failed to ensure Nextcloud folder '{remote_path}': {e}")
        return False

def upload_to_nextcloud(jpeg_bytes, remote_path):
    """
    Uploads a byte stream (our compressed JPEG) to Nextcloud.
    This is modified to not require a file on Google Drive.
    """
    try:
        jpeg_bytes.seek(0)
        full_url = f"{NEXTCLOUD_HOSTNAME}/{remote_path.lstrip('/')}"
        response = requests.put(
            full_url,
            data=jpeg_bytes,
            auth=(NEXTCLOUD_USERNAME, NEXTCLOUD_PASSWORD),
            headers={'Content-Type': 'image/jpeg'}
        )
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Failed to upload to Nextcloud path '{remote_path}': {e}")
        return False

# ==============================================================================
# --- 4. CORE PROCESSING LOGIC (COMBINED) ---
# ==============================================================================

def compress_dng_to_jpeg_bytes(dng_bytes):
    """
    Compresses a DNG byte stream to a JPEG of a target size.
    Includes a two-stage search (coarse then fine) for the optimal quality.
    Returns the JPEG as a byte stream or None if the target size cannot be achieved.
    """
    try:
        dng_bytes.seek(0)
        with rawpy.imread(dng_bytes) as raw:
            rgb_image = raw.postprocess(use_camera_wb=True, output_bps=8)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        high_quality = -1 # Tracks the last quality that was > 5MB
        
        # --- Stage 1: Coarse search (steps of 5) ---
        for quality in range(95, 10, -5):
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encoded_image = cv2.imencode('.jpeg', bgr_image, encode_param)
            if not result: continue

            file_size_mb = len(encoded_image) / (1024 * 1024)

            if MIN_JPEG_SIZE_MB <= file_size_mb <= MAX_JPEG_SIZE_MB:
                logging.info(f"  > Compression success (coarse): Quality {quality} -> {file_size_mb:.2f} MB.")
                jpeg_buffer = BytesIO(encoded_image)
                jpeg_buffer.seek(0)
                return jpeg_buffer
            
            if file_size_mb > MAX_JPEG_SIZE_MB:
                high_quality = quality
            
            elif file_size_mb < MIN_JPEG_SIZE_MB:
                low_quality = quality
                # If we jumped from >5MB to <1MB, start fine-grained search
                if high_quality != -1:
                    logging.info(f"  > Coarse search overshot. Starting fine search between quality {low_quality}-{high_quality}...")
                    # --- Stage 2: Fine-grained search ---
                    for fine_quality in range(high_quality - 1, low_quality, -1):
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), fine_quality]
                        result, encoded_image = cv2.imencode('.jpeg', bgr_image, encode_param)
                        if not result: continue
                        
                        file_size_mb = len(encoded_image) / (1024 * 1024)
                        if MIN_JPEG_SIZE_MB <= file_size_mb <= MAX_JPEG_SIZE_MB:
                            logging.info(f"  > Compression success (fine): Quality {fine_quality} -> {file_size_mb:.2f} MB.")
                            jpeg_buffer = BytesIO(encoded_image)
                            jpeg_buffer.seek(0)
                            return jpeg_buffer
                
                # If fine-grained search fails or wasn't triggered, fail the compression
                logging.warning(f"  > Could not achieve target size. Last attempt at quality {quality} was {file_size_mb:.2f} MB.")
                return None

        # If the loop finishes and the file is still too large
        logging.warning(f"  > Could not achieve target size. File is still too large at lowest quality setting.")
        return None

    except Exception as e:
        logging.error(f"  > Failed during DNG compression: {e}")
        return None


def process_dng_file(gdrive_service, nextcloud_client, file_info, original_path):
    """
    Main ETL logic for a single DNG file, now with on-the-fly compression.
    """
    file_id = file_info['id']
    file_name = file_info['name']

    # 1. Download DNG from Google Drive
    try:
        request = gdrive_service.files().get_media(fileId=file_id)
        dng_bytes = BytesIO(request.execute())
    except Exception as e:
        logging.error(f"Failed to download {file_name} (ID: {file_id}): {e}")
        return ('download_error', file_name)

    # 2. Extract Metadata
    lat, lon, dt = get_lat_lon_and_datetime(dng_bytes)
    if not all([lat, lon, dt]):
        return ('no_metadata', file_name)

    # 3. Compress DNG to JPEG (New step)
    logging.info(f"Compressing {file_name}...")
    compressed_jpeg_bytes = compress_dng_to_jpeg_bytes(dng_bytes)
    if not compressed_jpeg_bytes:
        return ('compress_fail', file_name)

    # 4. Call TBE API for folder structure
    api_success = False
    api_data = {}
    api_failure_reason = "Unknown API error"
    try:
        if not TBE_API_KEY: raise ValueError("TBE_API_KEY secret is missing.")
        params = {'lat': lat, 'lng': lon}
        headers = {'Key': TBE_API_KEY}
        response = requests.get(TBE_API_URL, params=params, headers=headers, timeout=20)
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

    # 5. Determine destination path and upload
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
        
        if upload_to_nextcloud(compressed_jpeg_bytes, f"{target_subfolder_path}/{new_filename}"):
            return ('success', file_id)
        else:
            return ('upload_fail', file_name)
    else:
        logging.warning(f"Uploading '{file_name}' to failure folder. Reason: {api_failure_reason}")
        failure_path = f"{NEXTCLOUD_ROOT_PATH}/{FAILED_API_FOLDER_NAME}"
        if ensure_nextcloud_folder(nextcloud_client, failure_path):
            if upload_to_nextcloud(compressed_jpeg_bytes, f"{failure_path}/{new_filename}"):
                return ('api_fail_upload_success', file_id)
        return ('upload_fail', file_name)


def traverse_drive(service, folder_id, current_path=""):
    """Recursively traverses Google Drive, yielding DNG file info and path."""
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
            if not results: break
            items = results.get('files', [])
            for item in items:
                item_path = os.path.join(current_path, item['name']).replace("\\", "/")
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    yield from traverse_drive(service, item['id'], item_path)
                elif item['name'].lower().endswith('.dng'):
                    yield item, os.path.splitext(item_path)[0]
            page_token = results.get('nextPageToken')
            if not page_token: break
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
    media = MediaFileUpload(local_path, mimetype='text/plain', resumable=True)
    try:
        if file_id:
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            file_metadata = {'name': os.path.basename(local_path), 'parents': [folder_id]}
            file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            return file.get('id')
    except HttpError as error:
        logging.error(f"Could not upload log file to Drive: {error}")
    return file_id


def main():
    """Main function to run the combined processing script."""
    start_time = time.monotonic()

    creds, _ = default()
    gdrive_service = build('drive', 'v3', credentials=creds)
    nextcloud_client = get_nextcloud_client()

    if not nextcloud_client:
        logging.critical("Could not connect to Nextcloud. Aborting.")
        return

    try:
        log_file_id = find_log_file_on_drive(gdrive_service, LOG_FOLDER_ID, STATE_FILE)
        if log_file_id:
            download_log_from_drive(gdrive_service, log_file_id, STATE_FILE)
        else:
            logging.info(f"No existing log file found on Google Drive. A new one will be created.")

        processed_ids = load_processed_files()
        logging.info(f"Loaded {len(processed_ids)} previously processed file IDs.")

        logging.info(f"Starting traversal from GDrive folder: '{SOURCE_ROOT_FOLDER_ID}'")
        all_tasks = list(traverse_drive(gdrive_service, SOURCE_ROOT_FOLDER_ID))
        tasks_to_run = [task for task in all_tasks if task[0]['id'] not in processed_ids]
        logging.info(f"Found {len(all_tasks)} total DNG files. {len(tasks_to_run)} files remaining to process.")

        for task in tqdm(tasks_to_run, desc="Processing Files"):
            if time.monotonic() - start_time > MAX_RUNTIME_SECONDS:
                logging.warning(f"Runtime limit reached. Stopping script.")
                break

            file_info, original_path = task
            status, detail = process_dng_file(gdrive_service, nextcloud_client, file_info, original_path)

            if status in ['success', 'api_fail_upload_success']:
                with open(STATE_FILE, 'a') as f:
                    f.write(f"{detail}\n")
                # Update the log on drive after each successful process
                new_log_id = update_log_on_drive(gdrive_service, log_file_id, LOG_FOLDER_ID, STATE_FILE)
                if new_log_id:
                    log_file_id = new_log_id

        logging.info("✅ Processing complete.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in main: {e}", exc_info=True)
        # Final attempt to save the log file in case of a crash
        logging.info("Attempting to save final log file to Google Drive...")
        update_log_on_drive(gdrive_service, log_file_id, LOG_FOLDER_ID, STATE_FILE)


# Run the main function
if __name__ == "__main__":
    main()
