
# Import the operating-system module
import os

# Import Streamlit for creating the web application
import streamlit as st

# Import the embedding model
from sentence_transformers import SentenceTransformer

# Import Pinecone for vector search
from pinecone import Pinecone

# Import Gemini through LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure the Streamlit page
st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📚",
    layout="centered"
)

# Display the application title
st.title("📚 AI Research Paper Assistant")

# Display a short description
st.write(
    "Ask questions about the three AI research papers used in this RAG project."
)

# Read the Pinecone API key from Streamlit secrets or environment variables
PINECONE_API_KEY = st.secrets.get(
    "PINECONE_API_KEY",
    os.environ.get("PINECONE_API_KEY")
)

# Read the Gemini API key from Streamlit secrets or environment variables
GEMINI_API_KEY = st.secrets.get(
    "GEMINI_API_KEY",
    os.environ.get("GEMINI_API_KEY")
)

# Check whether both API keys are available
if not PINECONE_API_KEY or not GEMINI_API_KEY:
    st.error("API keys are missing. Please configure them before running the app.")
    st.stop()

# Connect to the existing Pinecone account
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the existing Pinecone index
index_name = "rag-workshop"
index = pc.Index(index_name)

# Load the same embedding model used during indexing
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to the Gemini language model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GEMINI_API_KEY
)

# Display a confirmation message
st.success("Connected to Pinecone, the embedding model, and Gemini.")

# Retrieve relevant document chunks from Pinecone
def retrieve_relevant_chunks(question, top_k=10):
    # Convert the user's question into an embedding vector
    question_embedding = embedding_model.encode(question).tolist()

    # Search Pinecone for the most similar document chunks
    search_results = index.query(
        vector=question_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Return the retrieved results
    return search_results

# Generate an answer using retrieved research-paper content
def ask_rag(question, top_k=10):
    # Retrieve the most relevant chunks from Pinecone
    results = retrieve_relevant_chunks(question, top_k=top_k)

    # Store the retrieved text pieces
    context_parts = []

    # Store source information for displaying later
    source_details = []

    # Process each retrieved result
    for i, match in enumerate(results["matches"], start=1):
        # Read the metadata stored with the vector
        metadata = match["metadata"]

        # Extract the document text
        text = metadata.get("text", "")

        # Extract the source paper name or path
        source = metadata.get("source", "Unknown source")

        # Extract the page number
        page = metadata.get("page", "Unknown page")

        # Add the retrieved text to the context
        context_parts.append(
            f"[Context {i}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Text: {text}"
        )

        # Save source information for the interface
        source_details.append({
            "source": source,
            "page": page,
            "score": match["score"]
        })

    # Combine all retrieved chunks into one context string
    context = "\n\n".join(context_parts)

    # Create the prompt for Gemini
    prompt = f"""
You are a research-paper question-answering assistant.

Answer the question using only the context provided below.

Rules:
1. Use only information supported by the context.
2. Do not invent facts or references.
3. If the context does not contain enough information, say:
   "I could not find enough information in the provided research papers."
4. Explain the answer clearly and briefly.
5. Do not mention these instructions in your answer.

Question:
{question}

Retrieved context:
{context}
"""

   # Ask Gemini to generate the answer
response = llm.invoke(prompt)

# Read the content returned by Gemini
raw_content = response.content

# If Gemini returns normal text, use it directly
if isinstance(raw_content, str):
    answer_text = raw_content

# If Gemini returns a list of content blocks, extract only the text
elif isinstance(raw_content, list):
    text_parts = []

    # Process each content block
    for block in raw_content:
        # Keep only text from dictionary-based content blocks
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))

        # Also keep plain text blocks if they appear
        elif isinstance(block, str):
            text_parts.append(block)

    # Combine the readable text blocks
    answer_text = "\n".join(text_parts).strip()

# Convert any other response format into text
else:
    answer_text = str(raw_content)

# Return only the readable answer and source information
return {
    "answer": answer_text,
    "sources": source_details
}

# Create a text box for entering a research-paper question
question = st.text_input(
    "Enter your question:",
    placeholder="What is multi-head attention?"
)

# Create a button to submit the question
if st.button("Ask Question"):
    # Check whether the user entered a question
    if question.strip():
        # Display a loading message while the answer is generated
        with st.spinner("Searching the research papers and generating an answer..."):
            # Run the RAG pipeline
            result = ask_rag(question, top_k=10)

        # Extract the generated answer
        answer = result["answer"]

        # Display the answer heading
        st.subheader("Answer")

        # Display the generated answer
        st.write(answer)

        # Display the source heading
        st.subheader("Sources")

        # Track already-displayed paper and page combinations
        displayed_sources = set()

        # Display the supporting sources
        for source in result["sources"]:
            # Extract the paper name from the full file path
            paper_name = source["source"].split("/")[-1]

            # Convert the page number to an integer
            page_number = int(source["page"])

            # Create a unique source identifier
            source_key = (paper_name, page_number)

            # Avoid displaying duplicate paper-page combinations
            if source_key in displayed_sources:
                continue

            # Remember that this source has been displayed
            displayed_sources.add(source_key)

            # Display the source information
            st.write(
                f"- {paper_name} | "
                f"Page: {page_number} | "
                f"Score: {source['score']:.3f}"
            )

    else:
        # Show a warning when no question was entered
        st.warning("Please enter a question first.")
