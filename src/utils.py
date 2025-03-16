from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf_files(data):
    loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

#Split data into multiple chunks


def text_split(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(documents)

#Download embedding from huggingface

def download_hugging_face_embedding():
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings