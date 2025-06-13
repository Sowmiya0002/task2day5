import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import tempfile

# --- Fixed Gemini API Key ---
GOOGLE_API_KEY = "AIzaSyBoI2dqMaHAr3iwiQaW_-H_Jo9uAUxPqv4"

# --- Streamlit UI ---
st.set_page_config(page_title="Ask Your PDF 💬", layout="wide")
st.title("📄 Ask Your PDF using Google Gemini + RAG")
st.markdown("Upload a PDF file and ask questions based on its content.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# --- Text Input ---
question = st.text_input("Ask a question about the PDF:")
submit = st.button("🔍 Answer Me")

if uploaded_file and submit and question:
    try:
        # Step 1: Read PDF Text
        pdf_reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in pdf_reader.pages:
            raw_text += page.extract_text()

        if not raw_text:
            st.error("❌ Could not extract text from the PDF.")
            st.stop()

        # Step 2: Split Text
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(raw_text)

        # Step 3: Embedding Setup
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        db = FAISS.from_texts(texts, embedding=embeddings)

        # Step 4: Create Retriever
        retriever = db.as_retriever()

        # Step 5: Create Prompt Template
        prompt_template = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Use the following pieces of context to answer the question.
        Context: {context}
        Question: {question}
        Only return helpful answers based on the context.
        """)

        # Step 6: Set up Chat Model
        model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY
        )

        # Step 7: Build RAG Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            chain_type="stuff"
        )

        # Step 8: Run and Display Result
        response = qa_chain.run(question)
        st.success("✅ Answer generated successfully!")
        st.write(response)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
