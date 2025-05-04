import streamlit as st
import os
import tempfile
from pathlib import Path
import whisper
import openai

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Page Configuration
st.set_page_config(page_title="Thesis Agentic AI", layout="wide")
st.title("Thesis Writing Assistant")

# Step 1: OpenAI API Key Input
api_key = st.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key

# Step 2: Upload MP4s and PDFs
st.markdown("### Upload Instruction Videos (.mp4)")
mp4_files = st.file_uploader("Upload MP4 videos", type="mp4", accept_multiple_files=True)

st.markdown("### Upload Research Papers (.pdf)")
pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Storage path for vector database
db_dir = "data/db_streamlit"
Path(db_dir).mkdir(parents=True, exist_ok=True)

# Step 3: Transcribe Videos using Whisper
transcripts = []
if mp4_files:
    st.markdown("### Transcribing Videos")
    model = whisper.load_model("base")
    for mp4 in mp4_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(mp4.read())
            tmp_path = tmp.name
        result = model.transcribe(tmp_path)
        transcripts.append(result["text"])
        st.success(f"Transcribed: {mp4.name}")

# Step 4: Embed PDFs into Chroma Vector Store
docs = []
if pdf_files:
    st.markdown("### Embedding Research Papers")
    for pdf in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load_and_split())

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=db_dir
    )
    vectordb.persist()
    st.success(f"Embedded {len(pdf_files)} PDF(s)")

# Step 5: Retrieval-Augmented Generation (RAG) Prompt
if api_key and pdf_files:
    st.markdown("### Ask a Thesis-related Question")
    question = st.text_input("Your question")

    if question:
        vectordb = Chroma(
            persist_directory=db_dir,
            embedding_function=OpenAIEmbeddings()
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o"),
            retriever=vectordb.as_retriever()
        )
        answer = qa_chain.run(question)
        st.markdown("### Answer")
        st.write(answer)
