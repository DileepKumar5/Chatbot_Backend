import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone
from embeddings import get_vector_store
from extract_pdf import extract_pdf_content

# ✅ Load environment variables safely
load_dotenv()

# ✅ Ensure LangSmith API Key and Project are Set
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if not langsmith_api_key:
    raise ValueError("❌ ERROR: `LANGSMITH_API_KEY` is missing. Check `.env` file or set it as an environment variable.")

langsmith_project = os.getenv("LANGSMITH_PROJECT", "default_project")  # ✅ Set default project if missing

# ✅ Set environment variables explicitly
os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
os.environ["LANGSMITH_PROJECT"] = langsmith_project

# ✅ Log to check if API key and project are loaded correctly
print(f"✅ Loaded LangSmith API Key: {langsmith_api_key[:6]}****")
print(f"✅ Using LangSmith Project: {langsmith_project}")

# ✅ Initialize OpenAI Chat Model (GPT-4 or GPT-3.5-turbo)
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL"),
)

# ✅ Get Pinecone Vector Store
vector_store = get_vector_store()

def clean_and_humanize(text: str) -> str:
    """
    Cleans and structures retrieved text dynamically to improve readability.
    Completely removes Markdown symbols, tables, unnecessary characters.
    """
    if not text:
        return "No content available to process."

    # ✅ Remove all Markdown symbols
    text = re.sub(r"\*\*|\_\_", "", text)  # Remove bold/italic markers
    text = re.sub(r"\*+", "", text)  # Remove any remaining asterisks

    # ✅ Remove headers (e.g., ### Summary)
    text = re.sub(r"^#+\s*", "", text)  # Remove leading "###"

    # ✅ Remove table formatting (e.g., `| Header | Data |`)
    text = re.sub(r"\|", " ", text)  # Replace pipe (`|`) characters

    # ✅ Remove unnecessary brackets and extra spaces
    text = re.sub(r"[\[\]\n]", " ", text)  # Remove square brackets and newlines
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    text = re.sub(r"\([^)]+\)", "", text)  # Remove unnecessary parentheses

    # ✅ Remove trailing punctuation (avoids "Summary:" issue)
    text = re.sub(r"[:\-]+$", "", text).strip()

    # ✅ Improved prompt to GPT for **clean and human-friendly** responses
    prompt = f"""
    Convert the following extracted text into a clean, readable response:
    - Remove unnecessary formatting, symbols, and markdown.
    - Structure the response naturally, without special characters.

    **Extracted Text:**
    "{text}"
    """

    try:
        ai_response = llm.invoke(prompt)  # Get AIMessage object
        final_output = ai_response.content.strip()

        # ✅ Final Cleanup on GPT Output
        final_output = re.sub(r"\*\*|\_\_", "", final_output)  # Remove any missed bold text
        final_output = re.sub(r"\|", " ", final_output)  # Remove pipes if GPT used them
        final_output = re.sub(r"\s+", " ", final_output).strip()  # Normalize spaces

        return final_output
    except Exception as e:
        return f"Error processing response: {str(e)}"

def retrieve_answer_and_reference(query: str):
    """
    Dynamically retrieves and structures both the chatbot's answer and the reference answer from stored PDFs.
    """

    # Check if query is a simple greeting or casual input
    if query.lower() in ["hi", "hello", "hey", "howdy", "greetings"]:
        return "Hello! How can I assist you today?"

    try:
        retriever = vector_store.as_retriever()

        # ✅ Retrieve top 5 documents based on query relevance
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant reference found.", "The system couldn't find a matching answer. Please try rephrasing your question."

        # ✅ Extract the most relevant passage dynamically based on the query
        best_match = ""
        for doc in retrieved_docs[:5]:  # ✅ Scan top 5 results
            if any(word.lower() in doc.page_content.lower() for word in query.split()):
                best_match = doc.page_content
                break

        # ✅ If no good match found, use the highest-ranked document
        reference_answer = best_match if best_match else retrieved_docs[0].page_content

        # ✅ Clean and humanize the retrieved reference content
        refined_response = clean_and_humanize(reference_answer)

        return reference_answer, refined_response

    except Exception as e:
        return f"Error retrieving reference: {str(e)}", f"Error retrieving chatbot answer: {str(e)}"
