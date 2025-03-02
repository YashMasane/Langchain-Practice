from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

template = ChatPromptTemplate(
    [
    ('system', 'you are a helpful assistant in field {field}'),
    ('human', 'Give me information about {topic}')
    ]
)

temp = template.invoke(
    {'field': 'cricket',
     'topic': 'batting'}
)

print(temp)