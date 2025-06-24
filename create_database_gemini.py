# Library imports
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# gemini import
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
import google.generativeai as genai
import tiktoken

import os
import shutil
import random

# Load enviroment variables
# - You must create a .env file, with your own GOOGLE_API_KEY
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
enc = tiktoken.encoding_for_model("text-embedding-3-large")

CHROMA_PATH = "chroma-gemini"
DATA_PATH = "data"

# Parameters for chunk features in text splitting
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# Parameters for token limits
INDIVIDUAL_TOKEN_LIMIT = 8192
MAX_TOKENS = 300000
MAX_ITEMS = 2048


def main():
    """
    Main function to generate data for the database. 
    """
    generate_data()

def generate_data():
    """
    Generate data for the databse. 
    - documents are loaded from 'data' directory
    - documents are split into chunks
    - chunks are saved to a Chroma database
    """
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents() -> list[Document]:
    """
    Load documents from the 'data' directory

    :return: list of Document objects, each being a markdown file from 'data' directory
    """
    document_loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = document_loader.load()
    
    return documents

def split_text(documents: list[Document]) -> list[Document]:
    """
    Split documents into chunks through RecursiveCharacterTextSplitter
    - chunk_size: size of each chunk
    - chunk_overlap: overlap between chunks
    - length_function: function to calculate lenght of each chunk
    - add_start_index: whether to add start index to each chunk

    :param documents: list of Document objects obtained through DirectoryLoader
    :return: list of Document objects, each being a chunk of text from the original documents
    """
    # create a text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        length_function = len,          # len: counts number of characters in a string
        add_start_index = True
    )

    # split each document into chunks, print out number of documents, chunks
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # print out a random chunk and its metadata
    random_document_number = random.randint(0, len(chunks) - 1)
    document = chunks[random_document_number]
    print("--- Random chunk and metadata ---")
    print(f"Random chunk: {document.page_content}")
    print(f"Chunk metadata: {document.metadata}")
    print("---------------------------------")


    # return the 2d list of chunks
    return chunks

def batches(documents: list[Document]):
    """
    Create batches of documents based on token limits and maximum items.

    :param documents: list of Document objects, each being a chunk of text from the original documents
    :return: generator yielding batches of Document objects"""
    batch = []
    token_count = 0

    # iterate through documents and create batches based on token limits
    for doc in documents:
        tokens = len(enc.encode(doc.page_content))
        if tokens > INDIVIDUAL_TOKEN_LIMIT:
            print(f"Skipping document with {tokens} tokens, exceeds individual limit of {INDIVIDUAL_TOKEN_LIMIT}")
            continue
        # add document to batch if it doesn't exceed the token limit
        if token_count + tokens > MAX_TOKENS or len(batch) + 1 > MAX_ITEMS:
            yield batch
            batch = []
            token_count = 0
        # add document to batch
        batch.append(doc)
        token_count += tokens
    
    if batch:
        yield batch



def save_to_chroma(chunks: list[Document]):
    """
    Save chunks to a Chroma database. 

    :param chunks: 2d list of Document objects, each being a chunk of text from the original documents
    """
    # clear out the Chroma databse if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # create a new Chroma database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=GoogleGenerativeAIEmbeddings()
    )

    # add documents to the database in batches
    for batch in enumerate(batches(chunks), 1):
        db.add_documents(batch)

    # persist the database to disk
    # db.persist()      # Not needed, as Chroma does this automatically with persist_directory
    print(f"Saved {db._collection.count()} chunks to Chroma database at {CHROMA_PATH}")

if __name__ == "__main__":
    main()

