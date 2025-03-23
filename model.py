import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

def get_openai_response( api_key):
    os.environ["OPENAI_API_KEY"] = api_key  # Use your OpenAI API key
    try: 
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
        return llm
    except Exception as e:
        return f"Error: {e}"
    
def get_chatgroq_response(api_key):
    os.environ["GROQ_API_KEY"] = api_key
     
    try:
        llm=ChatGroq(model="qwen-2.5-32b",temperature=0.5)        
        return llm
    except Exception as e:
        return f"Error: {e}"