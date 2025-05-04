import streamlit as st
import os
import tempfile
import subprocess
import sys
from pathlib import Path
import time
import whisper
import openai
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup paths
DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
VIDEO_SUMMARIES_DIR = DATA_DIR / "video_summaries"
DOCUMENTS_DIR = DATA_DIR / "documents"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"

# Create necessary directories
for directory in [DATA_DIR, AUDIO_DIR, VIDEO_SUMMARIES_DIR, DOCUMENTS_DIR, KNOWLEDGE_BASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="Agentic RAG System",
    layout="wide",
)

# Check for API key
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""

# Sidebar for API key settings
with st.sidebar:
    st.title("🤖 Agentic RAG System")
    api_key = st.text_input("OpenAI API Key", value=st.session_state["OPENAI_API_KEY"], type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = api_key
        st.session_state["OPENAI_API_KEY"] = api_key
        st.success("API Key set successfully!")
    else:
        st.warning("Please enter your OpenAI API Key")
    
    st.divider()
    st.markdown("### About")
    st.markdown("""
    This application combines two agents:
    1. **Video Agent**: Transcribes and summarizes videos
    2. **Document Agent**: Provides RAG capabilities for PDF documents
    
    Both agents contribute to a unified knowledge base for research assistance.
    """)

# Function to transcribe and summarize video
def transcribe_and_summarize(video_path):
    st.info("Loading Whisper model...")
    model = whisper.load_model("base")
    
    st.info("Transcribing video...")
    result = model.transcribe(video_path)
    transcript = result["text"]
    
    st.info("Summarizing transcript with OpenAI...")
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Summarize this for thesis writing expectations:\n\n{transcript}"}]
    )
    summary = response.choices[0].message["content"]
    
    return transcript, summary

# Function to process PDF and create a QA tool
def load_pdf_qa_tool(pdf_path):
    st.info(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    st.info(f"Creating embeddings for {len(chunks)} document chunks...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return Tool(
        name="PDF Research Retriever",
        func=RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4o"),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        ).run,
        description="Retrieves answers from uploaded research PDFs."
    )

# Function to update knowledge base with new content
def update_knowledge_base(file_path, content, is_summary=False):
    filename = Path(file_path).stem
    if is_summary:
        output_path = KNOWLEDGE_BASE_DIR / f"{filename}_summary.txt"
    else:
        output_path = KNOWLEDGE_BASE_DIR / f"{filename}.txt"
    
    with open(output_path, "w") as f:
        f.write(content)
    
    return output_path

# Function to build RAG system from knowledge base
@st.cache_resource
def build_knowledge_base_rag():
    if not any(KNOWLEDGE_BASE_DIR.glob("*.txt")):
        return None
    
    st.info("Building RAG system from knowledge base...")
    documents = []
    
    # Load all text files in the knowledge base
    for text_file in KNOWLEDGE_BASE_DIR.glob("*.txt"):
        loader = TextLoader(text_file)
        documents.extend(loader.load())
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore

# Function to query the knowledge base
def query_knowledge_base(query):
    vectorstore = build_knowledge_base_rag()
    
    if vectorstore is None:
        return "Knowledge base is empty. Please add some documents or videos first."
    
    # Create retrieval QA chain
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa.run(query)

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Video Agent", "Document Agent", "Query Knowledge Base"])

# Tab 1: Video Agent
with tab1:
    st.header("Video Transcription & Summarization")
    
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.getvalue())
            video_path = tmp_file.name
        
        if st.button("Transcribe and Summarize"):
            if not api_key:
                st.error("Please set your OpenAI API Key in the sidebar first.")
            else:
                with st.spinner("Processing video..."):
                    try:
                        transcript, summary = transcribe_and_summarize(video_path)
                        
                        # Save results
                        transcript_path = AUDIO_DIR / f"{uploaded_video.name}_transcript.txt"
                        summary_path = VIDEO_SUMMARIES_DIR / f"{uploaded_video.name}_summary.txt"
                        
                        with open(transcript_path, "w") as f:
                            f.write(transcript)
                            
                        with open(summary_path, "w") as f:
                            f.write(summary)
                        
                        # Add to knowledge base
                        kb_path = update_knowledge_base(uploaded_video.name, summary, is_summary=True)
                        
                        # Display results
                        st.success("Video processed successfully!")
                        
                        st.subheader("Transcript")
                        st.text_area("Full Transcript", transcript, height=200)
                        
                        st.subheader("Summary")
                        st.text_area("Summary for Thesis", summary, height=200)
                        
                        st.success(f"Added to knowledge base: {kb_path}")
                        
                        # Clear the cache to rebuild the knowledge base
                        build_knowledge_base_rag.clear()
                    
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
                    
                    finally:
                        # Clean up the temporary file
                        os.unlink(video_path)

# Tab 2: Document Agent
with tab2:
    st.header("Document Processing & QA")
    
    uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_pdf is not None:
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_pdf.getvalue())
            pdf_path = tmp_file.name
        
        if st.button("Process Document"):
            if not api_key:
                st.error("Please set your OpenAI API Key in the sidebar first.")
            else:
                with st.spinner("Processing document..."):
                    try:
                        # Create QA tool
                        pdf_tool = load_pdf_qa_tool(pdf_path)
                        
                        # Extract document information
                        loader = PyPDFLoader(pdf_path)
                        documents = loader.load()
                        
                        # Create a summary of the document
                        combined_text = "\n\n".join([doc.page_content for doc in documents])
                        
                        # Summarize with OpenAI
                        response = openai.ChatCompletion.create(
                            model="gpt-4o",
                            messages=[{
                                "role": "user", 
                                "content": f"Create a comprehensive summary of this document for academic research purposes:\n\n{combined_text[:8000]}"
                            }]
                        )
                        summary = response.choices[0].message["content"]
                        
                        # Save to files
                        summary_path = DOCUMENTS_DIR / f"{uploaded_pdf.name}_summary.txt"
                        with open(summary_path, "w") as f:
                            f.write(summary)
                        
                        # Add to knowledge base
                        kb_path = update_knowledge_base(uploaded_pdf.name, summary, is_summary=True)
                        
                        # Display summary
                        st.success("Document processed successfully!")
                        st.subheader("Document Summary")
                        st.text_area("Summary", summary, height=300)
                        
                        st.success(f"Added to knowledge base: {kb_path}")
                        
                        # Allow specific questions about the document
                        st.subheader("Ask Questions About This Document")
                        doc_question = st.text_input("Enter your question:")
                        
                        if doc_question and st.button("Ask"):
                            answer = pdf_tool.func(doc_question)
                            st.subheader("Answer")
                            st.write(answer)
                        
                        # Clear the cache to rebuild the knowledge base
                        build_knowledge_base_rag.clear()
                    
                    except Exception as e:
                        st.error(f"Error processing document: {e}")
                    
                    finally:
                        # Clean up the temporary file
                        os.unlink(pdf_path)

# Tab 3: Query Knowledge Base
with tab3:
    st.header("Query Combined Knowledge Base")
    
    # Check if knowledge base exists
    kb_files = list(KNOWLEDGE_BASE_DIR.glob("*.txt"))
    
    if kb_files:
        st.info(f"Knowledge base contains {len(kb_files)} entries.")
        
        # Display available knowledge base entries
        with st.expander("View Knowledge Base Contents"):
            for file in kb_files:
                st.write(f"- {file.name}")
        
        # Query interface
        query = st.text_area("Enter your research question:", height=100)
        
        if query and st.button("Search Knowledge Base"):
            if not api_key:
                st.error("Please set your OpenAI API Key in the sidebar first.")
            else:
                with st.spinner("Searching knowledge base..."):
                    try:
                        answer = query_knowledge_base(query)
                        
                        st.subheader("Research Answer")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Error querying knowledge base: {e}")
    else:
        st.warning("Knowledge base is empty. Please process some videos or documents first.")

# Footer
st.markdown("---")
st.markdown("Agentic RAG System  | Built with Streamlit, Whisper, LangChain & OpenAI")
