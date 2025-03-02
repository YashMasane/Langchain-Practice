# message placeholders

from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model='deepseek-r1-distill-llama-70b')

template = ChatPromptTemplate(
    [
        ('system', 'You are a helpful customer support agent'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{query}')
    ]
)

chat_history = []

with open('chathistory.txt') as f:
    chat_history.extend(f.readlines())

prompt = template.invoke(
    {'chat_history': chat_history, 'query': 'Where is my refund'}
)

print(prompt)

