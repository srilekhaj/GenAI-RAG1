
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

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings #chatmodel, embedding model 
from dotenv import load_dotenv
load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01"  # As per documentation
)
# Ensure you provide the correct endpoint
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # Ensure this is correctly set

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
   azure_endpoint= azure_endpoint,
   api_key= os.getenv("AZURE_OPENAI_API_KEY"),
   azure_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
   
   #this is the template which works correctly for embedding keep variable name as same 
)



def get_page_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdfreader = PdfReader(pdf)
        for pagenum in pdfreader.pages:
            text += pagenum.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint= azure_endpoint,
        api_key= os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
    )
   #this is the template which works correctly for embedding keep variable name as same 
    
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")
    

def get_conversational_chain():
    PROMPT_TEMPLATE = """Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n 
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    #llm = AzureChatOpenAI(
     #   azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
     #   api_version= "2024-02-01",
     #   temperature= 0.3
    
    #)
 
    
    #prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    #chain = RetrievalQA(llm = llm, chain_type="stuff", retriever = retriever,  return_source_documents=True)
    #chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    #chain = llm.invoke(prompt=prompt.format("context"=context, "question"=question))
    #chain = chain.content
    
    return PROMPT_TEMPLATE


##########################

def user_input(user_question):
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint= azure_endpoint,
        api_key= os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
    )
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)
    
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
        
    with st.sidebar:
        st.title("Upload a PDF document and enter some text below.")
        pdf_docs = st.file_uploader("Upload PDF Document",  accept_multiple_files=True)
        
        if st.button("Submit"):
            with st.spinner("Reading PDF..."):
                raw_text = get_page_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
    


#st.title("Hello, Streamlit!")
#st.write("This is a simple Streamlit app.")