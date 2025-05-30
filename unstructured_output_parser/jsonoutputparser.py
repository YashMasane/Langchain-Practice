# json output parser example, returns data into json format

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGroq(model='deepseek-r1-distill-llama-70b')

# json output parser
output_parser = JsonOutputParser()

template = PromptTemplate(
    template='Give some information about following topic\n {topic}\n {format_instructions}',
    input_variables=['topic'],
    partial_variables = {'format_instructions': output_parser.get_format_instructions()}
)

chain = template | model | output_parser

result = chain.invoke({'topic': 'black holes'})

print(result)

