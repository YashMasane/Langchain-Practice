# getting structured output from opensource models
# here we have used string output parser to get string output from llm

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model='deepseek-r1-distill-llama-70b')

template1 = PromptTemplate(
    template='Write a detailed report on topic {topic}',
    input=['topic']
)

template2 = PromptTemplate(
    template='Write a 5 line summary on following text\n {text}',
    input=['text']
)

output_parser = StrOutputParser()

chain = template1 | model | output_parser | template2 | model | output_parser
result = chain.invoke({'topic': 'black hole'})

print(result)