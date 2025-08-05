# rag_chain.py
# A Retrieval-Augmented Generation (RAG) system that allows users to ask questions about a PDF document.
# Author: Muhammad Taha

import os
import uuid
import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

# === 🔐 Load Environment Variables ===
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# === 📄 Load and Split PDF ===
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return docs, text_splitter.split_documents(docs)

# === 🤖 Embedding Model ===
def initialize_embeddings():
    model = "BAAI/bge-large-en-v1.5"
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

# === 💬 LLM Setup ===
def initialize_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        max_completion_tokens=4000,
        temperature=0.1,
    )

# === 🔁 Question Rephrasing Chain ===
def initialize_rephrase_chain(llm):
    system_prompt = """
    Based on the chat history and the latest user message, rephrase the message as a standalone question. 
    Ensure the reformulated question is clear, self-contained, and preserves the original intent, 
    even if it refers to prior context. Do not answer the question—only rephrase it.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{question}")]
    )
    return prompt | llm | StrOutputParser()

# === 📖 RAG Chain ===
def initialize_rag_chain(llm):
    system_prompt = """
    You are a highly intelligent and accurate question-answering RAG (Retreival Augmented Generation) system.
    Your task is to answer the user's question based on the provided context.

    Instructions:
    1. Read the context carefully.
    2. Answer the question based on the context.
    3. If the context does not provide enough information, answer with your own knowledge but state that clearly.
    4. Do not make assumptions beyond the provided context.
    5. Provide a concise and accurate answer based on facts.

    Context: {context}
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{question}")]
    )
    return prompt | llm | StrOutputParser()

# === 🔍 Retrieve Context ===
def extract_context(query, file_uuid, embedding):
    qdrant_client = QdrantClient(url=QDRANT_URL)
    vector_store = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding,
    )
    condition = models.FieldCondition(
        key="metadata.file_uuid",
        match=models.MatchValue(value=file_uuid)
    )
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"filter": models.Filter(should=condition), "k": 5, "score_threshold": 0.0},
    )
    relevant_docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in relevant_docs])

# === 🚀 MAIN PROGRAM ===
def main():
    # Ask user for file path
    pdf_path = input("\nEnter the path to your PDF file: ").strip()
    if not os.path.exists(pdf_path):
        print("❌ File not found.")
        return

    # Initialize models
    embedding = initialize_embeddings()
    llm = initialize_llm()
    rephrase_chain = initialize_rephrase_chain(llm)
    rag_chain = initialize_rag_chain(llm)

    # Load and store docs
    docs, splitted_docs = load_and_split_pdf(pdf_path)
    file_uuid = str(uuid.uuid4())
    for doc in splitted_docs:
        doc.metadata.update({
            "user_name": "Muhammad Taha",
            "file_uuid": file_uuid,
            "num_pages": len(docs)
        })

    Qdrant.from_documents(
        documents=splitted_docs,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
    )

    # RAG Loop
    chat_history = []
    try:
        while True:
            user_question = input("\nAsk a question from the PDF (Ctrl+C to quit): ").strip()
            if not user_question:
                continue

            context = extract_context(user_question, file_uuid, embedding)
            rephrased_question = rephrase_chain.invoke({
                "chat_history": chat_history,
                "question": user_question,
            })

            print("\n💬 Answer:\n")
            response = ""
            for token in rag_chain.stream({
                "chat_history": chat_history,
                "question": rephrased_question,
                "context": context,
            }):
                print(token, end="", flush=True)
                response += token

            chat_history.append(HumanMessage(content=user_question))
            chat_history.append(AIMessage(content=response))

    except KeyboardInterrupt:
        print("\n\n🛑 Session ended by user.")


if __name__ == "__main__":
    main()
