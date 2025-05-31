from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model='llama3-8b-8192')

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Give a detail information about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate five point summary for following text\n {text}',
    input_variables=['text']
)

chain = prompt1 | model | parser | prompt2 | parser

result = chain.invoke({'topic': 'Cricket'})

print(result)

chain.get_graph().print_ascii()