
import os
from openai import AzureOpenAI
#from langchain.chat_models import AzureChatOpenAI 
from langchain_community.chat_models import AzureChatOpenAI

#as per documentation
from langchain_openai import AzureOpenAIEmbeddings
#  pip install langchain langchain-openai openai   
#langchain = framework
#openai - to use openai feature in AzureOpenAI
#langchain-openai = to use AzureOpenAIEmbeddings

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01"  # As per documentation
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


# Print the chat response
print(response.choices[0].message.content)
