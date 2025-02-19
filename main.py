from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
from google_drive import fetch_and_process_drive_files
from retrieval import retrieve_answer
from pydantic import BaseModel

# Define the body structure for the query input
class QueryRequest(BaseModel):
    query: str

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/fetch-drive-files/")
async def fetch_drive_files(background_tasks: BackgroundTasks):
    """Fetches files from Google Drive and processes them asynchronously."""
    try:
        background_tasks.add_task(fetch_and_process_drive_files)
        return {"message": "Processing Google Drive files in the background"}
    except Exception as e:
        logger.error(f"Error fetching drive files: {e}")
        raise HTTPException(status_code=500, detail="Failed to process Google Drive files")

@app.post("/query/")
async def query_chatbot(query_request: QueryRequest):
    """Retrieves answer from stored document embeddings."""
    try:
        query = query_request.query  # Get the query from the request body
        response = retrieve_answer(query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error retrieving answer: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve response")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
