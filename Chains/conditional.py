from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama3-8b-8192')

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give sentiment of feedback')

parser = PydanticOutputParser(pydantic_object=Feedback)
strparser = StrOutputParser()

template1 = PromptTemplate(
    template='Give the sentiment for following feedback\n{feedback}\n{format_information}',
    input_variables=['feedback'],
    partial_variables={'format_information': parser.get_format_instructions()}
)

classifier_chain = template1 | model | parser

print(classifier_chain.invoke({'feedback': 'This phone is terrible'}))

template2 = PromptTemplate(
    template='write an apropriate response for following positive feedback\n{feedback}',
    input_variables=['feedback']
)

template3 = PromptTemplate(
    template='write an apropriate response for following negative feedback\n{feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment=='positive', template2 | model | strparser),
    (lambda x: x.sentiment=='negative', template3 | model | strparser),
    RunnableLambda(lambda x: 'could not find sentiment')
    )

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'This phone is terrible'})

print(result)
