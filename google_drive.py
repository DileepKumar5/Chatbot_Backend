from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import os
import logging
from dotenv import load_dotenv
from embeddings import process_and_store_embeddings, document_exists_in_pinecone
import base64  # ✅ Add this at the top
import json
# Load environment variables
load_dotenv()

# Load Google Drive API credentials
SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")
# service_account_data = base64.b64decode(os.getenv("GOOGLE_SERVICE_ACCOUNT")).decode("utf-8")

FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Authenticate with Google Drive
try:
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    # creds = service_account.Credentials.from_service_account_info(
    #     json.loads(service_account_data), scopes=SCOPES
    # )
    drive_service = build("drive", "v3", credentials=creds)
    logger.info("✅ Successfully connected to Google Drive API")
except Exception as e:
    logger.error(f"❌ Failed to authenticate with Google Drive API: {e}")
    raise

# Allowed file types
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # PPTX
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  # XLSX
}

def list_files():
    """Lists supported files in the specified Google Drive folder."""
    try:
        if not FOLDER_ID:
            logger.error("❌ GOOGLE_DRIVE_FOLDER_ID is missing. Please check your .env file.")
            return []

        logger.info(f"📂 Checking files in Google Drive folder: {FOLDER_ID}")
        query = f"'{FOLDER_ID}' in parents and trashed=false"
        results = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        files = results.get("files", [])

        # Filter only allowed file types
        filtered_files = [f for f in files if f["mimeType"] in ALLOWED_MIME_TYPES]

        logger.info(f"📂 Found {len(filtered_files)} supported files in Google Drive")
        return filtered_files

    except Exception as e:
        logger.error(f"❌ Error listing files: {e}")
        return []

def download_file(file_id, file_name):
    """Downloads a file from Google Drive only if embeddings don't exist in Pinecone."""
    try:
        exists = document_exists_in_pinecone(file_name)
        logger.info(f"🔍 Checking if {file_name} exists in Pinecone: {exists}")

        if exists:
            logger.info(f"⏭️ Skipping {file_name}, embeddings already exist")
            return None

        request = drive_service.files().get_media(fileId=file_id)
        file_path = os.path.join("downloaded_docs", file_name)

        os.makedirs("downloaded_docs", exist_ok=True)

        with open(file_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

        logger.info(f"⬇️ Successfully downloaded: {file_name}")
        return file_path

    except Exception as e:
        logger.error(f"❌ Failed to download {file_name}: {e}")
        return None

def fetch_and_process_drive_files():
    """Fetches all supported files from Google Drive and processes embeddings **only if they don't exist**."""
    logger.info("🔄 Fetching files from Google Drive...")

    files = list_files()
    if not files:
        logger.warning("⚠️ No files found to process")
        return {"message": "No files found"}

    processed_files = 0
    for file in files:
        file_path = download_file(file["id"], file["name"])

        if file_path:
            logger.info(f"🧠 Processing embeddings for {file['name']}...")
            try:
                process_and_store_embeddings(file_path, file["name"])
                processed_files += 1
                logger.info(f"✅ Successfully processed {file['name']}")
            except Exception as e:
                logger.error(f"❌ Error processing embeddings for {file['name']}: {e}")

    logger.info(f"✅ Processing completed. {processed_files} new files processed.")
    return {"message": f"{processed_files} files processed successfully"}
