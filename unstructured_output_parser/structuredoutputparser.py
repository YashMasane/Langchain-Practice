# json output parser example, returns data into json format

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatGroq(model='deepseek-r1-distill-llama-70b')

schema = [ResponseSchema(name = 'fact1', description='fact 1 about the topic'),
          ResponseSchema(name = 'fact2', description='fact 2 about the topic'),
          ResponseSchema(name = 'fact3', description='fact 3 about the topic')]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give some information about following topic\n {topic}\n {format_instructions}',
    input_variables=['topic'],
    partial_variables = {'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic': 'black holes'})

print(result)

