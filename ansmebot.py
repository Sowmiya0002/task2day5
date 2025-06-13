import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnablePick
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import os

# --- Hardcoded Gemini API Key ---
GOOGLE_API_KEY = "AIzaSyBoI2dqMaHAr3iwiQaW_-H_Jo9uAUxPqv4"

# --- Streamlit UI Config ---
st.set_page_config(page_title="PDF Q&A with Gemini", layout="wide")
st.title("📄 Ask Your PDF using Google Gemini + RAG")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading and processing your PDF..."):
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load and chunk document
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Embed and store using FAISS
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        retriever = vectorstore.as_retriever()

        # Confirm to user
        st.success("✅ File uploaded and processed successfully. Now ask your question!")

        # Question Input
        question = st.text_input("Ask a question based on the uploaded PDF:")

        if question:
            with st.spinner("Thinking..."):
                # Create Prompt
                prompt = ChatPromptTemplate.from_template(
                    """
                    You are a helpful assistant. Answer the question using the provided context only.
                    Show the answer clearly and mention the page number if available.
                    Also give a short topic overview from that page.

                    Context:
                    {context}

                    Question:
                    {question}
                    """
                )

                # LLM
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-lite",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2
                )

                # RAG Chain
                rag_chain = (
                    RunnableMap({
                        "context": lambda x: retriever.invoke(x["question"]),
                        "question": RunnablePick(key="question")
                    })
                    | prompt
                    | llm
                )

                # Execute
                response = rag_chain.invoke({"question": question})

                # Display Answer
                st.markdown("### 📌 Answer:")
                st.write(response.content)

                # Optional Debug (context)
                with st.expander("🔍 Retrieved Chunks (Debug)"):
                    for doc in retriever.invoke(question):
                        st.write(f"**Page:** {doc.metadata.get('page', 'N/A')} — {doc.page_content[:200]}...")

        # Cleanup temp file
        os.remove(tmp_path)
