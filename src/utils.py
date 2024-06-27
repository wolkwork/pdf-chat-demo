import pathlib
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
import os

VECTORSTORE_PATH = "/app/data/vectorstore"
PDFS_PATH = "/app/data/pdfs/"


# def get_pdf_text(pdf_files):

#     text = ""
#     for pdf_file in pdf_files:
#         print(f"Processing {pdf_file}")
#         reader = PdfReader(pdf_file)
#         for page in reader.pages:
#             text += page.extract_text()
#     return text


# def get_chunk_text(text):

#     text_splitter = CharacterTextSplitter(
#         separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len
#     )

#     chunks = text_splitter.split_text(text)
#     print(f"Number of Chunks: {len(chunks)}")
#     return chunks


# def get_vector_store(text_chunks=[]):
#     # For OpenAI Embeddings

#     if os.path.isdir(VECTORSTORE_PATH) and not os.listdir(VECTORSTORE_PATH):
#         vectorstore: FAISS = FAISS.load_local(VECTORSTORE_PATH)
#         if text_chunks:
#             vectorstore.add_texts(texts=text_chunks, embedding=embeddings)
#             vectorstore.save_local(VECTORSTORE_PATH)
#     else:
#         vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


def get_llm():
    llm = ChatOpenAI(model="gpt-4o")
    return llm


def get_retriever():

    pdf_files = [
        filepath.absolute() for filepath in pathlib.Path(PDFS_PATH).glob("**/*.pdf")
    ]
    raw_documents = [
        doc for pdf_file in pdf_files for doc in PyPDFLoader(pdf_file).load()
    ]

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=VECTORSTORE_PATH
    )
    retriever = vectorstore.as_retriever()
    return retriever
