from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model='deepseek-r1-distill-llama-70b')

chat_history = [SystemMessage(content='You are a helpful assistant')]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))

    if user_input.lower() == 'exit':
        break
    result = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI: ', result.content)