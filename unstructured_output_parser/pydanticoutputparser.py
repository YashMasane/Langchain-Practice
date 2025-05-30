from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model='deepseek-r1-distill-llama-70b')

class Person(BaseModel):

    name: str = Field(description= 'Name of the person')
    age : int = Field(gt=18, description='Age of the person')
    city: str = Field(description='City where the person lives')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate name, age and city for a person who lives in {country}\n {format_instructions}",
    input_variables=['country'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

# prompt = template.invoke({'country': 'India'})
# print(prompt)

# result = model.invoke(prompt)

# result = parser.invoke(result.content)

chain = template | model | parser
result = chain.invoke({'country': 'India'})
print(result)