import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone
from embeddings import get_vector_store
<<<<<<< HEAD

# ✅ Load environment variables
=======
from extract_pdf import extract_pdf_content

# ✅ Load environment variables safely
>>>>>>> heroku/main
load_dotenv()

# ✅ Ensure LangSmith API Key and Project are Set
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if not langsmith_api_key:
<<<<<<< HEAD
    raise ValueError("❌ ERROR: `LANGSMITH_API_KEY` is missing.")

langsmith_project = os.getenv("LANGSMITH_PROJECT", "default_project")
=======
    raise ValueError("❌ ERROR: `LANGSMITH_API_KEY` is missing. Check `.env` file or set it as an environment variable.")

langsmith_project = os.getenv("LANGSMITH_PROJECT", "default_project")  # ✅ Set default project if missing
>>>>>>> heroku/main

# ✅ Set environment variables explicitly
os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
os.environ["LANGSMITH_PROJECT"] = langsmith_project

<<<<<<< HEAD
# ✅ Initialize OpenAI Chat Model
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL"),
    temperature=0.7,  # ✅ Set at initialization
    top_p=0.95,
    max_tokens=512  # ✅ Use `max_tokens` instead of `top_k`
=======
# ✅ Log to check if API key and project are loaded correctly
print(f"✅ Loaded LangSmith API Key: {langsmith_api_key[:6]}****")
print(f"✅ Using LangSmith Project: {langsmith_project}")

# ✅ Initialize OpenAI Chat Model (GPT-4 or GPT-3.5-turbo)
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL"),
>>>>>>> heroku/main
)

# ✅ Get Pinecone Vector Store
vector_store = get_vector_store()

def clean_and_humanize(text: str) -> str:
<<<<<<< HEAD
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
=======
    """
    Cleans and structures retrieved text dynamically to improve readability.
    Completely removes Markdown symbols, tables, unnecessary characters.
    """
    if not text:
        return "No content available to process."

    # Remove Markdown symbols, tables, and other unnecessary characters
    text = re.sub(r"\*\*|\_\_", "", text)  # Remove bold/italic markers
    text = re.sub(r"\*+", "", text)  # Remove any remaining asterisks
    text = re.sub(r"^#+\s*", "", text)  # Remove headers like "### Summary"
    text = re.sub(r"\|", " ", text)  # Replace pipe (`|`) characters from tables
    text = re.sub(r"[\[\]\n]", " ", text)  # Remove square brackets and newlines
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    text = re.sub(r"\([^)]+\)", "", text)  # Remove unnecessary parentheses
    text = re.sub(r"[:\-]+$", "", text).strip()  # Remove trailing punctuation

    # Clean up based on AI response
    prompt = f"""
    Convert the following extracted text into a clean, readable response:
    - Remove unnecessary formatting, symbols, and markdown.
    - Structure the response naturally, without special characters.
>>>>>>> heroku/main

    **Extracted Text:**
    "{text}"
    """

    try:
<<<<<<< HEAD
        ai_response = llm.predict(prompt)  # ✅ FIXED: Use `.predict()` instead of `.invoke()`
        return ai_response.strip()

=======
        ai_response = llm.invoke(prompt)  # Get AIMessage object
        final_output = ai_response.content.strip()

        # Final Cleanup on GPT Output
        final_output = re.sub(r"\*\*|\_\_", "", final_output)  # Clean any missed formatting
        final_output = re.sub(r"\|", " ", final_output)  # Remove pipes if GPT used them
        final_output = re.sub(r"\s+", " ", final_output).strip()  # Normalize spaces

        return final_output
>>>>>>> heroku/main
    except Exception as e:
        return f"Error processing response: {str(e)}"

def retrieve_answer_and_reference(query: str):
<<<<<<< HEAD
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
=======
    """
    Dynamically retrieves and structures both the chatbot's answer and the reference answer from stored PDFs.
    """
    try:
        retriever = vector_store.as_retriever()

        # Retrieve relevant documents based on the query
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant reference found.", "The system couldn't find a matching answer. Please try rephrasing your question."

        # Process the best match from the retrieved documents
        best_match = ""
        for doc in retrieved_docs[:5]:  # Look at top 5 results
            if any(word.lower() in doc.page_content.lower() for word in query.split()):
                best_match = doc.page_content
                break

        reference_answer = best_match if best_match else retrieved_docs[0].page_content

        # Clean and format the reference content
        refined_response = clean_and_humanize(reference_answer)

        # GPT to handle the response naturally based on the context
        prompt = f"""
        Based on the following content, please respond to the user's query as naturally and helpfully as possible:

        **Content:**
        "{refined_response}"

        **User Query:**
        "{query}"
        """

        ai_response = llm.invoke(prompt)  # Let GPT formulate the response

        # Return only the GPT response
        return refined_response, ai_response.content.strip()

    except Exception as e:
        return f"Error retrieving reference: {str(e)}", f"Error retrieving chatbot answer: {str(e)}"
>>>>>>> heroku/main
