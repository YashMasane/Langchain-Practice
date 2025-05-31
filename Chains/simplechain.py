from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model='llama3-8b-8192')

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Give five interesting facts about {topic}',
    input_variables=['topic']
)

chain = prompt | model | parser

result = chain.invoke({'topic': 'Cricket'})

print(result)

chain.get_graph().print_ascii()