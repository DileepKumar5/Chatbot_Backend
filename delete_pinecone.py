from pinecone import Pinecone
import os

# Load Pinecone API Key
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone Client
pc = Pinecone(api_key=pinecone_api_key)

# Delete Entire Index
pc.delete_index(pinecone_index_name)

print(f"✅ Successfully deleted Pinecone index: {pinecone_index_name}")
