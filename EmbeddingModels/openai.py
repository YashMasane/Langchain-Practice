from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

result = embeddings.embed_query("Mumbai is the capital of Maharashtra and it is financial capital\
                                of Indai also")

print(str(result))