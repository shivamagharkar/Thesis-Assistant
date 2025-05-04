from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def load_pdf_qa_tool(pdf_path: str) -> Tool:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    return Tool(
        name="PDF Research Retriever",
        func=RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4o"),
            retriever=retriever
        ).run,
        description="Retrieves answers from uploaded research PDFs."
    )

if __name__ == "__main__":
    pdf_path = "your_thesis_docs.pdf"  # Replace with your file
    pdf_tool = load_pdf_qa_tool(pdf_path)

    agent = initialize_agent(
        tools=[pdf_tool],
        llm=ChatOpenAI(model_name="gpt-4o"),
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

    query = "Summarize how fairness is handled differently in GenAI recommender systems."
    response = agent.run(query)

    print("\n--- Thesis Summary Output ---\n")
    print(response)
