import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# ---- Gemini API Key (fix here directly) ----
GOOGLE_API_KEY = "AIzaSyAg3Avl_PpsKX9yHS2HO_omaezsk5qvmwE"  # Replace with your real key

# ---- Streamlit UI ----
st.set_page_config(page_title="RAG PDF QA with Gemini", layout="centered")
st.title("📄 Ask Questions from Your PDF (Gemini + RAG)")
st.markdown("Upload a PDF, and ask anything based on its content!")

# ---- Upload PDF ----
pdf = st.file_uploader("📎 Upload your PDF file", type=["pdf"])

if pdf:
    try:
        # Read PDF text
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Split text into chunks
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_text(text)

        # Embedding and Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vectorstore = FAISS.from_texts(texts, embeddings)
        retriever = vectorstore.as_retriever()

        # Gemini LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)

        # Prompt Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using only the context provided from the PDF."),
            ("user", "{question}")
        ])

        # Chain: Prompt + LLM + Retriever
        qa_chain: Runnable = prompt | llm
        rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

        # ---- Question Input ----
        user_question = st.text_input("💬 Ask a question from the PDF")

        if st.button("Answer Me"):
            if user_question.strip() == "":
                st.warning("Please enter a question.")
            else:
                try:
                    answer = rag_chain.run(user_question)
                    st.success("✅ Answer generated!")
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"❌ Failed to generate answer: {e}")

    except Exception as e:
        st.error(f"❌ Failed to read PDF or initialize components: {e}")
