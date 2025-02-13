from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = 'Hii I am Yash, I am a data scientist and mathematician.'

result = embeddings.embed_query(text)

print(str(result))