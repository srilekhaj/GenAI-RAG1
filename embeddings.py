import streamlit as st

#st.title("Hello, Streamlit!")
#st.write("This is a simple Streamlit app.")
import os
from openai import AzureOpenAI
#from langchain_community.chat_models import AzureChatOpenAI

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# Initialize the Azure OpenAI client
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

# Generate a chat response
response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),  # Ensure this is set correctly
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
        {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
        {"role": "user", "content": "What are the services present in Azure AI services?"}
    ]
)



# Get the content of the last user message for embedding
user_query = "What are the services present in Azure AI services?"

# Generate the embedding for the user's query
single_vector = embeddings.embed_query(user_query)

# Print the embedding
print("Embedding vector:", single_vector)
print("First 100 elements of the embedding:", str(single_vector)[:100])


# Print the chat response
print(response.choices[0].message.content)
