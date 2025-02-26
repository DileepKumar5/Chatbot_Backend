import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone
from embeddings import get_vector_store

# ✅ Load environment variables
load_dotenv()

# ✅ Ensure LangSmith API Key and Project are Set
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if not langsmith_api_key:
    raise ValueError("❌ ERROR: `LANGSMITH_API_KEY` is missing.")

langsmith_project = os.getenv("LANGSMITH_PROJECT", "default_project")

# ✅ Set environment variables explicitly
os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
os.environ["LANGSMITH_PROJECT"] = langsmith_project

# ✅ Initialize OpenAI Chat Model
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL"),
)

# ✅ Get Pinecone Vector Store
vector_store = get_vector_store()

def clean_and_humanize(text: str) -> str:
    """Cleans retrieved text dynamically to improve readability."""
    if not text:
        return "No content available."

    text = re.sub(r"\*\*|\_\_", "", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"^#+\s*", "", text)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"[\[\]\n]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\([^)]+\)", "", text)
    text = re.sub(r"[:\-]+$", "", text).strip()

    prompt = f"""
    Convert the following extracted text into a clean, readable response:
    - Remove unnecessary formatting, symbols, and markdown.
    - Structure the response naturally.

    **Extracted Text:**
    "{text}"
    """

    try:
        ai_response = llm.invoke(prompt)

        # ✅ Ensure response format is valid
        if isinstance(ai_response, str):
            final_output = ai_response.strip()
        elif hasattr(ai_response, "content") and isinstance(ai_response.content, str):
            final_output = ai_response.content.strip()
        else:
            final_output = "Error: Unexpected response format."

        return final_output
    except Exception as e:
        return f"Error processing response: {str(e)}"

def retrieve_answer_and_reference(query: str):
    """Retrieves the best response by merging results from Pinecone & LLM."""
    try:
        retriever = vector_store.as_retriever()

        # ✅ Retrieve relevant documents based on the query
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            return {"response": "No relevant reference found."}

        # ✅ Process the best match from retrieved documents
        best_match = ""
        for doc in retrieved_docs[:5]:
            if hasattr(doc, "page_content") and any(word.lower() in doc.page_content.lower() for word in query.split()):
                best_match = doc.page_content
                break

        reference_answer = best_match if best_match else retrieved_docs[0].page_content

        # ✅ Clean and format the reference content
        refined_response = clean_and_humanize(reference_answer)

        # ✅ Generate a chatbot response using OpenAI
        prompt = f"""
        Based on the following retrieved content, respond to the user's query:

        **Context:**
        "{refined_response}"

        **User Query:**
        "{query}"
        """

        ai_response = llm.invoke(prompt)

        # ✅ Ensure `ai_response` is correctly formatted
        if isinstance(ai_response, str):
            final_response = ai_response.strip()
        elif hasattr(ai_response, "content") and isinstance(ai_response.content, str):
            final_response = ai_response.content.strip()
        else:
            final_response = "Error: Unexpected response format."

        return {"retrieved_context": refined_response, "response": final_response}

    except Exception as e:
        return {"retrieved_context": "Error retrieving context", "response": f"Error: {str(e)}"}
