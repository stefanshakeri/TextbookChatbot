# TextbookChatbot
A RAG OpenAI chatbot made to handle multiple textbooks as data. 

### Data
Under the 'data' folder, add markdown files for whatever data you'd like. This project was tested on various machine learning textbooks, so it's built to handle large amounts of data. 
### .env
Create your own .env file, and in it, include an 'OPENAI_API_KEY' like so: 
```
OPENAI_API_KEY='[enter key here]'
```
You can generate your own OpenAI key on their website: https://platform.openai.com/ 
### Running Files
Create and enter a virtual environment to use the requirements.txt file. 
To create the Chroma vector database: 
```
python create_database.py
```
To run a query: 
```
python query_data.py "[Your query here]"
```
