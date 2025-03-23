import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import os
 
from src.utils import  download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from src.prompt import *
import os


from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from model import get_chatgroq_response,get_openai_response

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Streamlit App UI
def main():
    # App title and description
    st.title("HCO Q&A Chatbot")
    st.write("Welcome to the HCO  Q&A chatbot !")

    # Sidebar for the API key input
    model=st.sidebar.selectbox("Model",["ChatGroq","ChatOpenAI"])
    st.sidebar.header("API Key Input")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    
    if model=="ChatOpenAI":
         st.sidebar.markdown("""
                    ### Need a OPEN API Key? 
                    If you don't have a OPEN API key, create api key  [ https://openai.com/index/openai-api/] 
                    """)
    else:
          st.sidebar.markdown("""
                    ### Need a Groq API Key?
                    If you don't have a Groq API key, create api key[ https://console.groq.com/keys] to get started.
                    """)
    
    os.environ["PINECONE_API_KEY"] = "pcsk_6d3uNY_NfSiPAVzwopGGrK1RWhw1RFyLu5gWPvXBtS4gHGTr6UAgDTwbgMZ7MWYs99DExZ"
    embeddings = download_hugging_face_embedding()

    docsearch = PineconeVectorStore.from_existing_index(index_name="hcobot",embedding=embeddings)

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
  
    
    # User's question input
    st.subheader("Ask a Question:")
    question = st.text_input("Your Question:")

    # If the user submits a question
    if st.button("Get Answer") and question :
        if model=="ChatOpenAI":
            if api_key:
                llm = get_openai_response(api_key)
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                answer = rag_chain.invoke({"input": question})
                st.subheader("Answer:")
                st.write(answer['answer'])
            else:
                st.sidebar.error("Please provide a valid OpenAI API key.")
                st.sidebar.markdown("""
                    ### Need a OPEN API Key?
                    If you don't have a OPEN API key, create [ https://openai.com/index/openai-api/] to get started.
                    """)
        else:
            if api_key:
                llm = get_chatgroq_response(api_key)
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                answer = rag_chain.invoke({"input": question})
                st.subheader("Answer:")
                st.write(answer['answer'])
            else:
                st.sidebar.error("Please provide a valid Groq API key.")
                st.sidebar.markdown("""
                    ### Need a Groq API Key?
                    If you don't have a Groq API key, create [ https://console.groq.com/keys] to get started.
                    """)
                

# Run the app
if __name__ == "__main__":
    main()



