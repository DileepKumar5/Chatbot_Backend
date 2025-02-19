import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # ✅ Correct Import
from langchain.chains import RetrievalQA
from embeddings import get_vector_store  # Import vector store from embeddings.py

# Load environment variables
load_dotenv()

# Initialize OpenAI LLM (GPT-4)
llm = ChatOpenAI(  # ✅ Use Correct Chat Model API
    api_key=os.getenv("OPENAI_API_KEY"),  
    model=os.getenv("OPENAI_MODEL")  # ✅ Ensure it's a chat model (GPT-4 or GPT-3.5-turbo)
)

# Get Pinecone Vector Store from embeddings.py
vector_store = get_vector_store()

def retrieve_answer(query: str):
    """ Retrieves relevant context and generates a response using GPT-4 """
    
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    # ✅ Invoke the chain correctly
    response = qa_chain.invoke({"query": query})

    # ✅ Return only the response text (fixes UI rendering issue)
    return response["result"] if isinstance(response, dict) and "result" in response else str(response)

