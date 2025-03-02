from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

document = [
    'Mumbai is the capital of Maharashtra',
    'Mumbai is the financial capital of India',
    'Nagpur is winter capital of Maharashtra',
]

# embedding for a single query
result = embeddings.embed_documents(document)

print(str(result))