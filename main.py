import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
from google_drive import fetch_and_process_drive_files
from retrieval import retrieve_answer_and_reference  # ✅ Correct Import
# from evaluation import evaluate_response_with_rag  # ✅ Evaluates chatbot answer vs reference
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input structures
class QueryRequest(BaseModel):
    query: str

class EvaluationRequest(BaseModel):
    query: str

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
    """Retrieves both the chatbot-generated answer and the reference answer from RAG system."""
    try:
        query = query_request.query
        reference, chatbot_response = retrieve_answer_and_reference(query)  # ✅ Retrieves both answers
        return {"response": chatbot_response}
    except asyncio.CancelledError:
        logger.warning("Request was cancelled.")
        raise HTTPException(status_code=499, detail="Request was cancelled by client.")
    except Exception as e:
        logger.error(f"Error retrieving answer: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve response")

# @app.post("/evaluate-rag/")
# async def evaluate_rag_response(evaluation_request: EvaluationRequest):
#     """Evaluates chatbot response from RAG system using NLP metrics."""
#     try:
#         query = evaluation_request.query

#         # ✅ Fetch the reference answer automatically and evaluate
#         evaluation_scores = evaluate_response_with_rag(query)

#         return evaluation_scores
#     except asyncio.CancelledError:
#         logger.warning("Request was cancelled.")
#         raise HTTPException(status_code=499, detail="Request was cancelled by client.")
#     except Exception as e:
#         logger.error(f"Error in evaluation: {e}")
#         raise HTTPException(status_code=500, detail="Failed to evaluate response")

if __name__ == "__main__":
    import uvicorn

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except asyncio.CancelledError:
        logger.warning("Server shutdown requested.")
