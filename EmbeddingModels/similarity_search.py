from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about virat'

document_embeddings = embeddings.embed_documents(documents)
query_embeddings = embeddings.embed_query(query)

similarity_score = cosine_similarity([query_embeddings], document_embeddings)[0]

index, _ = sorted(list(enumerate(similarity_score)), key= lambda x:x[1])[-1]

print(query)
print(documents[index])