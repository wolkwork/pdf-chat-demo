import hashlib
from pathlib import Path
from typing import IO, Union
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
import os

VECTORSTORE_PATH = "/app/data/vectorstore"
PDFS_PATH = "/app/data/pdfs/"


def get_llm():
    llm = ChatOpenAI(model="gpt-4o")
    return llm


def get_retriever():

    pdf_files = [filepath.absolute() for filepath in Path(PDFS_PATH).glob("**/*.pdf")]
    pdf_file_hash = hash_files(pdf_files)
    vectorstore_path = Path(VECTORSTORE_PATH)
    embeddings = OpenAIEmbeddings()
    # If we already made a vector db with these exact pdfs, load it
    if vectorstore_path.is_dir() and (vectorstore_path / pdf_file_hash).exists():
        print(f"Loading vectorstore from {vectorstore_path/pdf_file_hash}")
        vectorstore = Chroma(
            persist_directory=(vectorstore_path / pdf_file_hash).absolute().as_posix(),
            embedding_function=embeddings,
        )
        return vectorstore.as_retriever()

    raw_documents = [
        doc for pdf_file in pdf_files for doc in PyPDFLoader(pdf_file).load()
    ]
    print(f"Found {len(raw_documents)} documents in {len(pdf_files)} pdfs")

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len
    )
    documents = text_splitter.split_documents(raw_documents)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=(vectorstore_path / pdf_file_hash).absolute().as_posix(),
    )
    retriever = vectorstore.as_retriever()
    return retriever


def hash_files(file_paths: list[Path]) -> str:
    hash = hashlib.md5()
    for file_path in file_paths:
        if not Path(file_path).is_file():
            raise ValueError(f"File does not exist: {file_path}")
        hash.update(file_bytes(file_path))
    return hash.hexdigest()


def hash_fileIO(fileIO: IO[bytes]) -> str:
    hash = hashlib.md5()
    hash.update(fileIO_bytes(fileIO))
    return hash.hexdigest()


def file_bytes(file_path: Union[str, Path]) -> bytes:
    with open(file_path, "rb") as f:
        return fileIO_bytes(f)


def fileIO_bytes(fileIO: IO[bytes]) -> bytes:
    return fileIO.read()


def string_bytes(s: str) -> bytes:
    return s.encode("utf-8")
