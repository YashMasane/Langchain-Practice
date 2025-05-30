# getting structured output from opensource models

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='Write a detailed report on topic {topic}',
    input=['topic']
)

template2 = PromptTemplate(
    template='Write a 5 line summary on following text\n {text}',
    input=['text']
)

prompt1 = template1.invoke({'topic': 'data science'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})

result2 = model.invoke(prompt2)

print(result2)