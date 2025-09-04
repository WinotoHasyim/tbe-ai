# ==============================================================================
# --- 1. SETUP AND AUTHENTICATION FOR GOOGLE COLAB ---
# ==============================================================================

# Install necessary libraries and download ExifTool
!pip install pyexiftool webdavclient3 tqdm pandas rawpy opencv-python-headless numpy google-api-python-client google-auth-httplib2 google-auth-oauthlib > /dev/null
!wget -q -O Image-ExifTool-13.34.tar.gz https://sourceforge.net/projects/exiftool/files/Image-ExifTool-13.34.tar.gz/download
!tar -xzf Image-ExifTool-13.34.tar.gz
!mv -f Image-ExifTool-13.34/* .

import os
import logging
import requests
import time
import rawpy
import cv2
import json
import numpy as np
import pandas as pd
import exiftool
import subprocess
from io import BytesIO
from datetime import datetime, timezone
from webdav3 import client as wc

# --- Third-party libraries ---
from google.colab import auth
from google.auth import default
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload
from tqdm.notebook import tqdm

# --- Colab Specific Setup ---
auth.authenticate_user()
from google.colab import userdata

# ==============================================================================
# --- 2. CONFIGURATION ---
# ==============================================================================
# --- Google Drive Settings ---
SOURCE_ROOT_FOLDER_ID = userdata.get("SOURCE_ROOT_FOLDER_ID")
# LOG_FOLDER_ID = userdata.get("LOG_FOLDER_ID")

# # --- Nextcloud Settings (Destination) ---
# NEXTCLOUD_HOSTNAME = userdata.get("NEXTCLOUD_HOSTNAME")
# NEXTCLOUD_USERNAME = userdata.get("NEXTCLOUD_USERNAME")
# NEXTCLOUD_PASSWORD = userdata.get("NEXTCLOUD_PASSWORD")
# NEXTCLOUD_ROOT_PATH = userdata.get("NEXTCLOUD_ROOT_PATH")
# FAILED_API_FOLDER_NAME = "FAILED"

# # --- TBE API Settings ---
# TBE_API_URL = userdata.get("TBE_API_URL")
# TBE_API_KEY = userdata.get("TBE_API_KEY")

# --- State Management ---
MASTER_LOG_CSV = "master_log.csv"
MAX_RUNTIME_SECONDS = 11 * 3600 + 45 * 60


MONITORING_KEYWORDS = [
    "tikus",
    "tanaman",
    "ngengat",
    "ulat",
    "penggerek"
]
MIN_JPEG_SIZE_MB = 1.0
MAX_JPEG_SIZE_MB = 5.0

# --- Required Metadata Keys ---
REQUIRED_METADATA_KEYS = [
    "EXIF:GPSLatitude",
    "EXIF:GPSLatitudeRef",
    "EXIF:GPSLongitude",
    "EXIF:GPSLongitudeRef",
    "EXIF:GPSAltitude",
    "XMP:AbsoluteAltitude",
    "XMP:RelativeAltitude",
    "XMP:GimbalYawDegree",
    "XMP:GimbalPitchDegree",
    "XMP:GimbalRollDegree"
]

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


# ==============================================================================
# --- 3. HELPER FUNCTIONS (Fungsi Bantuan) ---
# ==============================================================================

def extract_all_metadata_with_exiftool(dng_byte_stream, temp_dng_path="temp.dng"):
    lat, lon, all_metadata = None, None, {}
    try:
        with open(temp_dng_path, "wb") as f:
            f.write(dng_byte_stream.getvalue())

        with exiftool.ExifToolHelper(executable="./exiftool") as et:
            all_metadata = et.get_metadata(temp_dng_path)[0]

        if "EXIF:GPSLatitude" in all_metadata and "EXIF:GPSLongitude" in all_metadata:
            lat = all_metadata["EXIF:GPSLatitude"]
            lon = all_metadata["EXIF:GPSLongitude"]
            if 'S' in all_metadata.get('EXIF:GPSLatitudeRef', 'N'): lat = -lat
            if 'W' in all_metadata.get('EXIF:GPSLongitudeRef', 'E'): lon = -lon
    except Exception as e:
        logging.error(f"Error extracting metadata with ExifTool: {e}")
    finally:
        if os.path.exists(temp_dng_path):
            os.remove(temp_dng_path)
    return lat, lon, all_metadata

def get_nextcloud_client():
    options = {'webdav_hostname': NEXTCLOUD_HOSTNAME, 'webdav_login': NEXTCLOUD_USERNAME, 'webdav_password': NEXTCLOUD_PASSWORD}
    try:
        client = wc.Client(options); client.verify = True; return client
    except Exception as e:
        logging.error(f"Failed to initialize Nextcloud client: {e}"); return None

def ensure_nextcloud_folder(client, remote_path):
    try:
        if not client.check(remote_path): client.mkdir(remote_path)
        return True
    except Exception as e:
        logging.error(f"Failed to ensure Nextcloud folder '{remote_path}': {e}"); return False

def upload_to_nextcloud(jpeg_bytes, remote_path):
    try:
        jpeg_bytes.seek(0)
        full_url = f"{NEXTCLOUD_HOSTNAME}/{remote_path.lstrip('/')}"
        response = requests.put(full_url, data=jpeg_bytes, auth=(NEXTCLOUD_USERNAME, NEXTCLOUD_PASSWORD), headers={'Content-Type': 'image/jpeg'})
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Failed to upload to Nextcloud path '{remote_path}': {e}"); return False

def find_log_file_on_drive(service, folder_id, filename):
    query = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
    try:
        response = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
        return response.get('files', [{}])[0].get('id')
    except (HttpError, IndexError) as e:
        logging.warning(f"Could not find log file {filename} in Drive: {e}"); return None

def download_log_from_drive(service, file_id, local_path):
    if not file_id: return False
    try:
        request = service.files().get_media(fileId=file_id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: status, done = downloader.next_chunk()
        with open(local_path, 'wb') as f: f.write(fh.getvalue())
        logging.info(f"Successfully downloaded log file from Drive to '{local_path}'.")
        return True
    except HttpError as error:
        logging.error(f"Could not download log file: {error}"); return False

def update_log_on_drive(service, file_id, folder_id, local_path):
    media = MediaFileUpload(local_path, mimetype='text/csv', resumable=True)
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

def list_files_with_retry(service, query, page_token):
    max_retries = 5; delay = 1
    for attempt in range(max_retries):
        try:
            return service.files().list(q=query, pageSize=1000, fields="nextPageToken, files(id, name, mimeType)", pageToken=page_token).execute()
        except HttpError as error:
            if error.resp.status in [500, 502, 503, 504]:
                logging.warning(f"API call failed with transient error {error.resp.status}. Retrying in {delay}s...")
                time.sleep(delay); delay *= 2
            else: raise
    logging.error(f"API call failed after {max_retries} retries for query: {query}"); return None

def traverse_drive(service, folder_id):
    # Use actual source folder name instead of 'ROOT'
    service2 = build('drive', 'v3', credentials=service._http.credentials)
    folder_info = service2.files().get(fileId=folder_id, fields='name').execute()
    source_folder_name = folder_info.get('name', 'ROOT')
    folders_to_scan = [(folder_id, source_folder_name)]
    while folders_to_scan:
        current_id, current_path = folders_to_scan.pop(0)
        query = f"'{current_id}' in parents and trashed = false"
        page_token = None
        while True:
            results = list_files_with_retry(service, query, page_token)
            if not results:
                logging.error(f"Could not list files in folder ID {current_id} after retries. Skipping folder."); break
            for item in results.get('files', []):
                item_path = os.path.join(current_path, item['name']).replace("\\", "/")
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    folders_to_scan.append((item['id'], item_path))
                elif item['name'].lower().endswith('.dng'):
                    yield item, os.path.splitext(item_path)[0]
            page_token = results.get('nextPageToken')
            if not page_token: break

# ==============================================================================
# --- 4. CORE PROCESSING LOGIC (Logika Inti) ---
# ==============================================================================

def compress_dng_to_jpeg_bytes(dng_bytes):
    """Hanya melakukan kompresi, tanpa menangani metadata EXIF."""
    try:
        dng_bytes.seek(0)
        with rawpy.imread(dng_bytes) as raw:
            rgb_image = raw.postprocess(use_camera_wb=True,output_bps=8)
            cam_wb = raw.camera_whitebalance  # numpy array [R, G1, B, G2]
            print(f"Camera White balance = {cam_wb}")

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        high_quality = -1

        for quality in range(95, 10, -5):
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encoded_image_data = cv2.imencode('.jpeg', bgr_image, encode_param)
            if not result: continue

            file_size_mb = len(encoded_image_data) / (1024 * 1024)
            if MIN_JPEG_SIZE_MB <= file_size_mb <= MAX_JPEG_SIZE_MB:
                logging.info(f"  > Compression success (coarse): Quality {quality} -> {file_size_mb:.2f} MB."); return BytesIO(encoded_image_data)

            if file_size_mb > MAX_JPEG_SIZE_MB:
                high_quality = quality
            elif file_size_mb < MIN_JPEG_SIZE_MB and high_quality != -1:
                low_quality = quality
                logging.info(f"  > Coarse search overshot. Starting fine search between {low_quality}-{high_quality}...")
                for fine_quality in range(high_quality - 1, low_quality, -1):
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), fine_quality]
                    result, encoded_image_data = cv2.imencode('.jpeg', bgr_image, encode_param)
                    if not result: continue
                    file_size_mb = len(encoded_image_data) / (1024 * 1024)
                    if MIN_JPEG_SIZE_MB <= file_size_mb <= MAX_JPEG_SIZE_MB:
                        logging.info(f"  > Compression success (fine): Quality {fine_quality} -> {file_size_mb:.2f} MB."); return BytesIO(encoded_image_data)
                logging.warning(f"  > Fine search failed."); return None
        logging.warning("  > Could not achieve target size."); return None
    except Exception as e:
        logging.error(f"  > Failed during DNG compression: {e}", exc_info=True); return None

def inject_metadata_with_exiftool(dng_bytes, jpeg_bytes, temp_dng="temp.dng", temp_jpeg="temp.jpeg"):
    """Menyuntikkan semua metadata dari DNG ke JPEG menggunakan ExifTool."""
    try:
        with open(temp_dng, "wb") as f:
            f.write(dng_bytes.getvalue())
        with open(temp_jpeg, "wb") as f:
            f.write(jpeg_bytes.getvalue())

        cmd = ["./exiftool", "-tagsFromFile", temp_dng, "-all:all", "-overwrite_original", temp_jpeg]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        with open(temp_jpeg, "rb") as f:
            final_jpeg_bytes = BytesIO(f.read())

        return final_jpeg_bytes
    except subprocess.CalledProcessError as e:
        logging.error(f"ExifTool failed to inject metadata: {e.stderr}")
        return None
    except Exception as e:
        logging.error(f"An error occurred during metadata injection: {e}")
        return None
    finally:
        if os.path.exists(temp_dng): os.remove(temp_dng)
        if os.path.exists(temp_jpeg): os.remove(temp_jpeg)

def process_and_upload_file(gdrive_service, nextcloud_client, file_info, original_gdrive_path):
    dng_id, original_filename = file_info['id'], file_info['name']
    log_data = {
        "dng_id": dng_id,
        "original_gdrive_path": original_gdrive_path,
        "original_filename": original_filename,
        "latitude": None,
        "longitude": None,
        "api_land_name": None,
        "api_land_id": None,
        "api_hst": None,
        "api_hss": None,
        "adjusted_hst": None,
        "adjusted_hss": None,
        "gdrive_uploaded_date": None,
        "hst_hss_negative": False,
        "api_response_json": None,
        "final_nextcloud_path": None,
        "status": "UNKNOWN_ERROR",
        "timestamp": datetime.now().isoformat(),
        "exif_data_json": None
    }

    try:
        # Get Google Drive uploaded date
        file_metadata = gdrive_service.files().get(fileId=dng_id, fields='createdTime').execute()
        gdrive_uploaded_date = file_metadata.get('createdTime', None)
        log_data["gdrive_uploaded_date"] = gdrive_uploaded_date
        request = gdrive_service.files().get_media(fileId=dng_id)
        dng_bytes = BytesIO(request.execute())
    except HttpError as e:
        logging.error(f"Failed to download {original_filename}: {e}"); log_data["status"] = "DOWNLOAD_FAIL"; return log_data

    lat, lon, all_exif_data = extract_all_metadata_with_exiftool(dng_bytes)
    missing_keys = [key for key in REQUIRED_METADATA_KEYS if key not in all_exif_data]
    if missing_keys:
        logging.warning(f"Skipping {original_filename} due to missing required metadata: {missing_keys}")
        log_data["status"] = "MISSING_METADATA_FAIL"
        log_data["exif_data_json"] = json.dumps(all_exif_data)
        return log_data
    if not all([lat, lon]):
        logging.warning(f"Skipping {original_filename} due to missing GPS metadata."); log_data["status"] = "NO_METADATA"; return log_data
    log_data.update({
        "latitude": lat,
        "longitude": lon,
        "exif_data_json": json.dumps(all_exif_data)
    })

    compressed_jpeg_bytes = compress_dng_to_jpeg_bytes(dng_bytes)
    if not compressed_jpeg_bytes:
        logging.error(f"Failed to compress {original_filename}. Skipping."); log_data["status"] = "COMPRESS_FAIL"; return log_data

    final_jpeg_bytes = inject_metadata_with_exiftool(dng_bytes, compressed_jpeg_bytes)
    if not final_jpeg_bytes:
        logging.error(f"Failed to inject metadata into {original_filename}. Skipping."); log_data["status"] = "METADATA_INJECT_FAIL"; return log_data

    new_filename_part = original_gdrive_path.replace('/', '_')
    try:
        params, headers = {'lat': lat, 'lng': lon}, {'Key': TBE_API_KEY}
        response = requests.get(TBE_API_URL, params=params, headers=headers, timeout=20)
        log_data["api_response_json"] = json.dumps(response.json())
        response.raise_for_status()

        if response.json().get('status') is True and response.json().get('data', {}).get('land_id'):
            api_data = response.json()['data']
            land_name = api_data.get('land_name', None)
            land_id = api_data['land_id']
            hst = int(api_data.get('land_hst', '0'))
            hss = int(api_data.get('land_hss', '0'))
            log_data["api_land_name"] = land_name
            log_data["api_land_id"] = land_id
            log_data["api_hst"] = hst
            log_data["api_hss"] = hss
            # Calculate days since upload
            days_since_upload = 0
            if gdrive_uploaded_date:
                try:
                    uploaded_dt = datetime.strptime(gdrive_uploaded_date, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
                    now_dt = datetime.now(timezone.utc)
                    days_since_upload = (now_dt - uploaded_dt).days
                except Exception as ex:
                    logging.warning(f"Could not parse gdrive_uploaded_date: {gdrive_uploaded_date}, error: {ex}")
            # Adjust hst/hss only if not zero
            adjusted_hst = hst - days_since_upload if hst != 0 else 0
            adjusted_hss = hss - days_since_upload if hss != 0 else 0
            log_data["adjusted_hst"] = adjusted_hst
            log_data["adjusted_hss"] = adjusted_hss
            if adjusted_hst < 0 or adjusted_hss < 0:
                log_data["hst_hss_negative"] = True
                logging.warning(f"Adjusted hst/hss is negative for file {original_filename}: adjusted_hst={adjusted_hst}, adjusted_hss={adjusted_hss}")
                log_data["status"] = "NEGATIVE_HST_HSS_SKIP"
                return log_data
            # Use adjusted value for foldering
            day_val = str(adjusted_hst) if adjusted_hst != 0 else str(adjusted_hss)
            main_folder_path = f"{NEXTCLOUD_ROOT_PATH}/{land_id}-{day_val}"
            if ensure_nextcloud_folder(nextcloud_client, main_folder_path):
                for keyword in MONITORING_KEYWORDS: ensure_nextcloud_folder(nextcloud_client, f"{main_folder_path}/monitoring {keyword}")
                target_subfolder = next((f"monitoring {kw}" for kw in MONITORING_KEYWORDS if kw in original_gdrive_path.lower()), "monitoring tanaman")
                final_path = f"{main_folder_path}/{target_subfolder}/{new_filename_part}.jpeg"
                log_data["final_nextcloud_path"] = final_path
                log_data["status"] = "SUCCESS" if upload_to_nextcloud(final_jpeg_bytes, final_path) else "UPLOAD_FAIL"
            else:
                log_data["status"] = "NEXTCLOUD_FOLDER_FAIL"
            return log_data
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for {original_filename}: {e}"); log_data["api_response_json"] = json.dumps({"error": str(e)})

    # Fallback untuk kegagalan API
    logging.warning(f"API call failed for {original_filename}. Uploading to FAILED folder.")
    log_data["api_land_id"] = "API_FAIL"
    failure_path = f"{NEXTCLOUD_ROOT_PATH}/{FAILED_API_FOLDER_NAME}"
    ensure_nextcloud_folder(nextcloud_client, failure_path)
    final_path = f"{failure_path}/{new_filename_part}.jpeg"
    log_data["final_nextcloud_path"] = final_path
    log_data["status"] = "API_FAIL_UPLOADED" if upload_to_nextcloud(final_jpeg_bytes, final_path) else "API_FAIL_UPLOAD_FAIL"
    return log_data

# ==============================================================================
# --- 5. MAIN EXECUTION (Eksekusi Utama) ---
# ==============================================================================
def main():
    # --- User: Only process a single DNG file by Drive file ID ---
    creds, _ = default()
    gdrive_service = build('drive', 'v3', credentials=creds)
    # Set your DNG file ID here
    specific_dng_id = "1fWLz3xufKT9YNnAHbzqLQetuPNFSoGtI"  # <-- Replace with your actual file ID

    # Find the file info for the given ID
    all_tasks = list(traverse_drive(gdrive_service, SOURCE_ROOT_FOLDER_ID))
    match = next(((fi, path) for fi, path in all_tasks if fi['id'] == specific_dng_id), None)
    if not match:
        logging.error(f"DNG file with ID {specific_dng_id} not found.")
        return
    file_info, original_gdrive_path = match
    logging.info(f"Processing DNG file: {file_info['name']} ({original_gdrive_path})")

    # Download DNG
    try:
        request = gdrive_service.files().get_media(fileId=specific_dng_id)
        dng_bytes = BytesIO(request.execute())
    except HttpError as e:
        logging.error(f"Failed to download {file_info['name']}: {e}")
        return

    # Compress DNG to JPEG
    compressed_jpeg_bytes = compress_dng_to_jpeg_bytes(dng_bytes)
    if not compressed_jpeg_bytes:
        logging.error("JPEG compression failed.")
        return

    # Save compressed JPEG to Colab storage for download
    jpeg_save_path = f"/content/{os.path.splitext(file_info['name'])[0]}_compressed.jpeg"
    compressed_jpeg_bytes.seek(0)
    with open(jpeg_save_path, "wb") as f:
        f.write(compressed_jpeg_bytes.read())
    logging.info(f"Compressed JPEG saved to: {jpeg_save_path}")

if __name__ == "__main__":
    main()

