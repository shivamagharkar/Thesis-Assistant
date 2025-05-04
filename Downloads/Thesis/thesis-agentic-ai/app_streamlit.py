import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import tempfile
from pathlib import Path
import whisper
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
docs = []
if pdf_files:
    st.markdown("###Embedding Research Papers")
    for pdf in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load_and_split())
    vectordb = Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        persist_directory=db_dir
    )
    vectordb.persist()
    st.success(f"Embedded {len(pdf_files)} PDF(s)")
if api_key and pdf_files:
    st.markdown("###Ask a Thesis-related Question")
    question = st.text_input("Your question")

    if question:
        vectordb = Chroma(
            persist_directory=db_dir,
            embedding_function=OpenAIEmbeddings()
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o-mini"),  # or "gpt-4" if you prefer
            retriever=vectordb.as_retriever()
        )
        answer = qa_chain.run(question)
        st.markdown("###Answer")
        st.write(answer)
