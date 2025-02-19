import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
import fitz  # PyMuPDF
import docx
import pandas as pd
from pptx import Presentation
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore  # ✅ Fixed Import
from langchain.text_splitter import RecursiveCharacterTextSplitter  # ✅ More efficient splitter
import hashlib  # ✅ Added to create unique chunk IDs

# Load environment variables
load_dotenv()

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Load Pinecone Index Name
index_name = os.getenv("PINECONE_INDEX_NAME")

# Check if Pinecone index exists, if not create it
if index_name not in pc.list_indexes().names():
    logger.info(f"🛠 Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=3072,  # ✅ Matches OpenAI embeddings
        metric="cosine"
    )

# Connect to Pinecone Index
index = pc.Index(index_name)

# Initialize OpenAI Embeddings
embedding_model = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))

# Set up Pinecone VectorStore
vector_store = PineconeVectorStore(index=index, embedding=embedding_model, namespace="default")

logger.info("✅ Pinecone & OpenAI Embeddings Initialized Successfully!")


def extract_text(file_path: str) -> str:
    """Extracts text from different file types with proper formatting."""
    try:
        if file_path.endswith(".pdf"):
            doc = fitz.open(file_path)
            text = "\n".join([page.get_text("text") for page in doc])

        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])

        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path, dtype=str)  # Convert all to string
            text = df.to_string(index=False, header=True)

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


def document_exists_in_pinecone(filename: str) -> bool:
    """Properly checks if embeddings for a file exist in Pinecone using metadata filtering."""
    try:
        # ✅ Query Pinecone for metadata matching the filename
        response = index.query(vector=[], filter={"source": filename}, top_k=1, include_metadata=True)

        # ✅ If matches are found, embeddings exist
        if response and response.get("matches"):
            logger.info(f"🔍 File '{filename}' already exists in Pinecone. Skipping processing.")
            return True  # File embeddings exist

        logger.info(f"📄 File '{filename}' not found in Pinecone. Needs processing.")
        return False  # No match found

    except Exception as e:
        logger.error(f"❌ Error checking Pinecone for {filename}: {e}")
        return False  # Assume it doesn't exist to avoid skipping new files




def generate_chunk_id(filename: str, chunk_text: str) -> str:
    """Generates a unique hash ID for each chunk based on filename and chunk text."""
    return hashlib.sha256(f"{filename}-{chunk_text}".encode()).hexdigest()


def process_and_store_embeddings(file_path: str, filename: str) -> int:
    """
    Processes document, extracts text, and stores embeddings only if not already stored.
    Uses RecursiveCharacterTextSplitter for better chunking.
    """
    # ✅ Check if embeddings for the specific file already exist
    if document_exists_in_pinecone(filename):
        logger.info(f"⏭️ Skipping {filename}, embeddings already exist in Pinecone")
        return 0  # Skip processing

    text = extract_text(file_path)

    if not text:
        logger.warning(f"⚠️ Skipping {filename} due to empty content")
        return 0  # Return 0 to indicate no chunks were stored

    # ✅ Improved text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)

    chunk_ids = [generate_chunk_id(filename, chunk) for chunk in chunks]
    metadata = [{"source": filename, "chunk_id": chunk_id, "chunk": i} for i, chunk_id in enumerate(chunk_ids)]

    # ✅ Store embeddings in Pinecone with unique IDs
    vector_store.add_texts(chunks, metadatas=metadata, ids=chunk_ids)
    logger.info(f"✅ Stored {len(chunks)} chunks for {filename} in Pinecone")

    return len(chunks)  # Return the number of stored chunks


def get_vector_store():
    """Returns the initialized Pinecone vector store for retrieval."""
    return vector_store
