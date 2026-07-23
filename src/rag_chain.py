import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from src.prompts import build_prompt

load_dotenv()

PERSIST_DIR = "chroma_db"

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )

def build_vectorstore(chunks, collection_name="doc_qa"):
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore

def load_vectorstore(collection_name="doc_qa"):
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

def get_llm(temperature=0.2):
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=temperature,
    )

def format_docs(docs):
    return "\n\n".join(
        f"[Source: page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs
    )

def build_rag_chain(vectorstore, persona="default"):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = build_prompt(persona)
    llm = get_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def build_conversational_chain(vectorstore, persona="default"):
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": build_prompt(persona, use_few_shot=False).partial(chat_history="")},
    )
    return chain, memory