import os
import streamlit as st
import pickle 
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()



st.title("News Research Tool")
st.sidebar.title("News Article URLs")
urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500) 

if process_url_clicked:
    loaders = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading Started")
    data = loaders.load()

    #split the data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size = 1000
    )
    main_placeholder.text("Text Splitter Started!!")
    docs = text_splitter.split_documents(data)

    # Embedding and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # method 1 for saving the embedding
    vectorstore.save_local("vectorstore")
    main_placeholder.text("Embedding Vector Started building")
    time.sleep(2)

    # method 2 for saving the embedding
    # with open(file_path, "wb") as f:
    #     pickle.dump(vectorstore, f)


query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result is a dictionary having two elements. Answer and source
            st.header("Answer")
            st.subheader(result["answer"])
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)