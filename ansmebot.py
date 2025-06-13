import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import tempfile
import os

# --- Set Google Gemini API Key ---
GOOGLE_API_KEY = "AIzaSyBoI2dqMaHAr3iwiQaW_-H_Jo9uAUxPqv4"

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Q&A with Gemini", layout="wide")
st.title("📚 Ask Questions from PDF using Gemini + RAG")
st.markdown("Upload a PDF and ask questions. Uses Google Gemini + LangChain RAG.")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    # --- Load & Split PDF ---
    try:
        loader = PyPDFLoader(tmp_pdf_path)
        pages = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(pages)

        # --- Generate Embeddings & Store in FAISS ---
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        # --- Setup LLM ---
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
        )

        # --- Custom Prompt Template ---
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Use the following context to answer the question.
        If the answer is not in the document, say so.

        Context:
        {context}

        Question:
        {input}
        """)

        # --- Create Chain ---
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

        # --- Question Input ---
        question = st.text_input("💬 Ask a question from the PDF:")

        if st.button("Get Answer"):
            if question:
                try:
                    result = rag_chain.invoke({"input": question})
                    st.success("✅ Answer:")
                    st.write(result["answer"])
                except Exception as e:
                    st.error(f"⚠️ Error during retrieval: {e}")
            else:
                st.warning("❗ Please enter a question.")

    except Exception as e:
        st.error(f"⚠️ Error loading PDF: {e}")
