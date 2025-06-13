import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyBoI2dqMaHAr3iwiQaW_-H_Jo9uAUxPqv4"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Streamlit app layout
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 Ask Questions from Your PDF")
st.markdown("Upload a PDF and ask questions using Google Gemini & FAISS-powered RAG pipeline.")

# Upload PDF
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None:
    # Save and load PDF
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    # Text splitting
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    # Embedding
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # User input
    query = st.text_input("Ask a question from your PDF:")

    if query:
        with st.spinner("Searching and generating answer..."):
            # RAG: Retrieve + Answer
            docs = vectorstore.similarity_search(query)
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)

            # Display
            st.subheader("💬 Answer:")
            st.success(response)

            with st.expander("🔍 Relevant Context"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}**: {doc.page_content[:500]}...")

