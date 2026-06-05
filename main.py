import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.6
)
st.title("InsightFetch")
st.subheader("Learn from any URL...")
main_placeholder = st.empty()
if st.sidebar.button("Remove URL") and len(st.session_state.urls) > 1:
    st.session_state.urls.pop()

if "urls" not in st.session_state:
    st.session_state.urls = [""]

if st.sidebar.button("Add URL"):
    st.session_state.urls.append("")

for i in range(len(st.session_state.urls)):
    st.session_state.urls[i] = st.sidebar.text_input(
        f"URL {i+1}",
        value=st.session_state.urls[i],
        key=f"url_{i}"
    )

if st.sidebar.button("Process URLs"):
    Loader = UnstructuredURLLoader(urls=st.session_state.urls)
    main_placeholder.info("Loading started....")
    documents = Loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Enter your question: ")
if query:
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    retriever = vectorstore.as_retriever()
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm = llm, retriever=retriever)
    result = chain({"question": query},return_only_outputs=True)
    st.header("Answer:")
    st.subheader(result["answer"])

