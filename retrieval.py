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
    temperature=0.7,
    top_p=0.95,
    max_tokens=512
)

# ✅ Get Pinecone Vector Store
vector_store = get_vector_store()


def clean_text(text: str) -> str:
    """Cleans text and ensures list formatting with new lines."""
    if not text:
        return "No content available."

    # ✅ Remove Markdown formatting (**Bold**, *Italic*, __Underlines__, etc.)
    text = re.sub(r"\*\*|\_\_", "", text)  # Remove bold and underline markers
    text = re.sub(r"\*", "", text)  # Remove remaining asterisks
    text = re.sub(r"\|", " ", text)  # Replace pipe (`|`) characters
    text = re.sub(r"\[[^\]]*\]", "", text)  # Remove text inside square brackets
    text = re.sub(r"\([^\)]*\)", "", text)  # Remove text inside parentheses
    text = re.sub(r"[:\-]+$", "", text)  # Remove trailing colons and dashes
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces

    # ✅ Ensure each list item is on a **new line** (Fixing improper inline formatting)
    text = re.sub(r"(\d+)\.\s", r"\n\1. ", text)  # Force new line before numbered items

    return text.strip()


def clean_and_humanize(text: str) -> str:
    """Cleans and structures retrieved text dynamically for better readability."""
    cleaned_text = clean_text(text)

    prompt = f"""
    Convert the following extracted text into a clean, readable response:
    - Remove unnecessary formatting, symbols, and markdown.
    - Ensure list items appear **on separate lines**.
    - Structure the response naturally.

    **Extracted Text:**
    "{cleaned_text}"
    """

    try:
        ai_response = llm.predict(prompt)
        return clean_text(ai_response.strip())  # ✅ Ensuring clean output

    except Exception as e:
        return f"Error processing response: {str(e)}"


def retrieve_answer_and_reference(query: str):
    """Retrieves the best response by first showing context, then generating an answer."""
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.get_relevant_documents(query)

        if not retrieved_docs:
            return {
                "retrieved_context": "No relevant context found.",
                "response": "I'm sorry, I couldn't find relevant information."
            }

        # ✅ Extract retrieved contexts
        contexts = [clean_text(doc.page_content.strip()) for doc in retrieved_docs if hasattr(doc, "page_content")]
        formatted_context = "\n\n---\n\n".join(contexts) if contexts else "No relevant context found."

        # ✅ LLM Generation with Tuned Parameters
        prompt = f"""
        **Context from Database:**
        {formatted_context}

        **User Query:**
        {query}

        **Instructions:**
        - Answer the question accurately using the provided context.
        - Ensure list items are presented **on new lines** for clarity.
        - If uncertain, state that the information is unavailable.
        """

        ai_response = llm.predict(prompt)
        final_response = clean_text(ai_response.strip())

        return {
            "retrieved_context": formatted_context,
            "response": final_response
        }

    except Exception as e:
        return {
            "retrieved_context": "Error retrieving context",
            "response": f"Error: {str(e)}"
        }
