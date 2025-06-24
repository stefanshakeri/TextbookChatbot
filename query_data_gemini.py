from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import google.genai as genai

import argparse
import os

# Load environment variables
CHROMA_PATH = "chroma-gemini"

# Load enviroment variables
# - You must create a .env file, with your own OPENAI_API_KEY 
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# prompt template for the question-answering task
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---

Answer the question based on the above context, going into as much detail as possible: {question}
If the answer is not in the context, say you don't know. 
"""

# number of chunks to return from the Chroma database when queried
RETURN_AMT = 4
# threshold for similarity score to consider a result relevant
SIMILARITY_THRESHOLD = 0.5

def get_query_text() -> str:
    """
    Generate a query text for the database. 
    
    :return: query text as a string
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    return query_text

def prepare_db() -> Chroma:
    """
    Prepare the database for querying.

    :return: Chroma database object
    """
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )
    return db

def main():

    # create CLI (command line interface)
    query_text = get_query_text()

    # prepare the database
    db = prepare_db()

    # search the database
    results = db.similarity_search_with_relevance_scores(query_text, k=RETURN_AMT)
    
    if len(results) == 0 or results[0][1] < SIMILARITY_THRESHOLD:
        print(f"length of results: {len(results)}")
        print(f"first result score: {results[0][1] if results else 'N/A'}")
        print("No relevant results found.")
        return

    # compile the reulting context from the documents into a single string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # create a prompt template and format it with the context + query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # intialize the model and get the response
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response_text = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )
    
    # print the response and the sources
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text.text}\n\nSources: {', '.join(sources)}"
    print(formatted_response)

if __name__ == "__main__":
    main()