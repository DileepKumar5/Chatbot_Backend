import os
import logging
import hashlib
from dotenv import load_dotenv
from pinecone import Pinecone
import fitz  # PyMuPDF for PDFs
import docx
import pandas as pd
from pptx import Presentation
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Load environment variables
load_dotenv()

# ✅ Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=pinecone_api_key)

# ✅ Check if Pinecone index exists, if not, create it
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    logger.info(f"🛠 Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=3072,  # Matches OpenAI embeddings
        metric="dotproduct"  # ✅ Changed to dot product similarity
    )

# ✅ Connect to Pinecone Index
index = pc.Index(index_name)

# ✅ Initialize OpenAI Embeddings
embedding_model = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))

# ✅ Set up Pinecone VectorStore with namespace support
vector_store = PineconeVectorStore(index=index, embedding=embedding_model, namespace="default")

logger.info("✅ Pinecone & OpenAI Embeddings Initialized Successfully!")

# ✅ Extract text from different file types
def extract_text(file_path: str) -> str:
    """Extracts text from PDF, DOCX, PPTX, and XLSX files with better handling."""
    try:
        if file_path.endswith(".pdf"):
            with fitz.open(file_path) as doc:
                text = "\n".join([page.get_text("text") for page in doc])

        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])

        elif file_path.endswith(".xlsx"):
            # Handling multiple sheets
            df_dict = pd.read_excel(file_path, sheet_name=None, dtype=str)  # Read all sheets into a dictionary
            text = ""
            for sheet_name, sheet_df in df_dict.items():
                text += f"\n\nSheet Name: {sheet_name}\n" + sheet_df.to_string(index=False, header=True)
            if not text.strip():
                logger.warning(f"⚠️ No text found in Excel file: {file_path}")
                return None

        elif file_path.endswith(".pptx"):
            prs = Presentation(file_path)
            text = "\n".join(
                [shape.text.strip() for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
            )

        else:
            raise ValueError(f"❌ Unsupported file type: {file_path}")

        if not text.strip():
            logger.warning(f"⚠️ No text found in file: {file_path}")
            return None

        return text

    except Exception as e:
        logger.error(f"❌ Error extracting text from {file_path}: {e}")
        return None

# ✅ Check if document exists in Pinecone before processing
def document_exists_in_pinecone(filename: str) -> bool:
    """Checks if embeddings for a file exist in Pinecone using metadata filtering."""
    try:
        response = index.query(vector=[], filter={"source": filename}, top_k=1, include_metadata=True)

        if response and response.get("matches"):
            logger.info(f"🔍 File '{filename}' already exists in Pinecone. Skipping processing.")
            return True  

        logger.info(f"📄 File '{filename}' not found in Pinecone. Needs processing.")
        return False  

    except Exception as e:
        logger.error(f"❌ Error checking Pinecone for {filename}: {e}")
        return False  

# ✅ Generate a unique chunk ID
def generate_chunk_id(filename: str, chunk_text: str) -> str:
    """Generates a unique hash ID for each chunk based on filename and chunk text."""
    return hashlib.sha256(f"{filename}-{chunk_text}".encode()).hexdigest()

# ✅ Process and store embeddings with improved chunking
def process_and_store_embeddings(file_path: str, filename: str) -> int:
    """
    Extracts text, chunks it efficiently, and stores embeddings in Pinecone.
    Uses RecursiveCharacterTextSplitter for optimal chunking.
    """
    if document_exists_in_pinecone(filename):
        logger.info(f"⏭️ Skipping {filename}, embeddings already exist in Pinecone")
        return 0  

    text = extract_text(file_path)

    if not text:
        logger.warning(f"⚠️ Skipping {filename} due to empty content")
        return 0  

    # ✅ Improved Chunking: Smaller Chunks with Overlap for Better Retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # ⬅️ Reduced chunk size for better segmentation
        chunk_overlap=128,  # ⬅️ Increased overlap for better context retention
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    chunk_ids = [generate_chunk_id(filename, chunk) for chunk in chunks]
    metadata = [{"source": filename, "chunk_id": chunk_id, "chunk_number": i} for i, chunk_id in enumerate(chunk_ids)]

    # ✅ Store embeddings in Pinecone with unique chunk IDs
    vector_store.add_texts(chunks, metadatas=metadata, ids=chunk_ids)
    logger.info(f"✅ Stored {len(chunks)} chunks for {filename} in Pinecone")

    return len(chunks)  

# ✅ Return the initialized Pinecone vector store
def get_vector_store():
    """Returns the initialized Pinecone vector store for retrieval."""
    return vector_store
