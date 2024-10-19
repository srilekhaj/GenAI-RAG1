
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document  # Import Document
from langchain.chains import RetrievalQA
import os
from openai import AzureOpenAI  #normal azureopenai model
#from langchain_community.chat_models import AzureChatOpenAI

from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01"  # As per documentation
)


def get_conversational_chain():
    PROMPT_TEMPLATE = """Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n 
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    
    return PROMPT_TEMPLATE


##########################

def user_input(user_question):
    
    #retriever = new_db.as_retriever() 
    chain = get_conversational_chain()
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version= "2024-02-01",
        temperature= 0.3
    )
    prompt = chain.format(context= docs, question= user_question)  #line no 98 #lineno 119
    response = llm.invoke(prompt)  #prompttemplate + embedding respone + the userquestion
    
    
    print(response.content) #cmd
    st.write(response.content)  #getting response in ui

# Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with PDFs using Azure OpenAI")
    
    user_question = st.chat_input("Ask a question ")
    
    if user_question:
        user_input(user_question)
        
