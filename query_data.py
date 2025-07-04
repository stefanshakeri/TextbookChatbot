from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import openai

import argparse
import os

# Load environment variables
CHROMA_PATH = "chroma"

# Load enviroment variables
# - You must create a .env file, with your own OPENAI_API_KEY 
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_MODEL_NAME = "gpt-4o"

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
SIMILARITY_THRESHOLD = 0.7

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
    embedding_function = OpenAIEmbeddings()
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
        print("No relevant results found.")
        return

    # compile the reulting context from the documents into a single string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # create a prompt template and format it with the context + query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # intialize the model and get the response
    model = ChatOpenAI(model_name=GPT_MODEL_NAME)
    response_text = model.invoke(prompt)

    # print the response and the sources
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text.content}\n\nSources: {', '.join(sources)}"
    print(formatted_response)

if __name__ == "__main__":
    main()