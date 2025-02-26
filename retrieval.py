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
    temperature=0.7,  # ✅ Set at initialization
    top_p=0.95,
    max_tokens=512  # ✅ Use `max_tokens` instead of `top_k`
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
        ai_response = llm.predict(prompt)  # ✅ FIXED: Use `.predict()` instead of `.invoke()`
        return ai_response.strip()

    except Exception as e:
        return f"Error processing response: {str(e)}"

def retrieve_answer_and_reference(query: str):
    """Retrieves the best response by first showing context, then generating an answer."""
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # ✅ Retrieve top 5 results
        retrieved_docs = retriever.get_relevant_documents(query)  # ✅ FIXED: Use `.get_relevant_documents(query)`

        if not retrieved_docs:
            return {
                "retrieved_context": "No relevant context found.",
                "response": "I'm sorry, I couldn't find relevant information."
            }

        # ✅ Extract retrieved contexts
        contexts = [doc.page_content.strip() for doc in retrieved_docs if hasattr(doc, "page_content")]
        formatted_context = "\n\n---\n\n".join(contexts) if contexts else "No relevant context found."

        # ✅ LLM Generation with Tuned Parameters
        prompt = f"""
        **Context from Database:**
        {formatted_context}

        **User Query:**
        {query}

        **Instructions:**
        - Answer the question accurately using the provided context.
        - If uncertain, state that the information is unavailable.
        - Avoid assumptions or fabricated responses.
        """

        ai_response = llm.predict(prompt)  # ✅ FIXED: Removed `top_k`
        final_response = ai_response.strip()

        return {
            "retrieved_context": formatted_context,
            "response": final_response
        }

    except Exception as e:
        return {
            "retrieved_context": "Error retrieving context",
            "response": f"Error: {str(e)}"
        }
