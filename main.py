import streamlit as st
from dotenv import load_dotenv
import boto3
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import BedrockChat
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from streamlit_cognito_auth import CognitoAuthenticator
from aws_secretsmanager_caching import SecretCache, SecretCacheConfig
import botocore.session
import json
import os

# Load environment variables
load_dotenv()

# Ensure the AWS region is set
os.environ['AWS_REGION'] = 'us-east-1'  # Replace 'us-east-1' with your desired region

# Debugging: Print environment variables to ensure AWS_REGION is set correctly
print("AWS_REGION:", os.getenv('AWS_REGION'))

# Debugging: Verify that the IAM role is being used
try:
    response = requests.get('http://169.254.169.254/latest/meta-data/iam/security-credentials/')
    role_name = response.text
    print(f"IAM Role Name: {role_name}")
    response = requests.get(f'http://169.254.169.254/latest/meta-data/iam/security-credentials/{role_name}')
    print(f"IAM Role Credentials: {response.json()}")
except Exception as e:
    print(f"Error accessing IAM role metadata: {e}")

# Create a botocore session and specify the region
try:
    botocore_session = botocore.session.get_session()
    client = botocore_session.create_client('secretsmanager', region_name=os.getenv('AWS_REGION'))
    cache_config = SecretCacheConfig()
    cache = SecretCache(config=cache_config, client=client)
    secret = json.loads(cache.get_secret_string('gmgf_secrets'))
except Exception as e:
    st.error(f"Error accessing Secrets Manager: {e}")
    st.stop()

# Set Streamlit page configuration
st.set_page_config(page_title='Green Mountain Girls Farm Instruction Manual Assistant')

# Initialize the CognitoAuthenticator
authenticator = CognitoAuthenticator(
    pool_id=secret['POOL_ID'],
    app_client_id=secret['CLIENT_ID'],
    app_client_secret=secret['CLIENT_SECRET'],
    use_cookies=True
)

# Check login status
is_logged_in = authenticator.login()
if not is_logged_in:
    st.stop()

def logout():
    print("Logout in example")
    authenticator.logout()

# ------------------------------------------------------
# Amazon Bedrock - settings
try:
    # Create a Boto3 session without specifying a profile
    session = boto3.Session(region_name=os.getenv('AWS_REGION'))
    bedrock_runtime = session.client(
        service_name="bedrock-runtime",
        region_name=os.getenv('AWS_REGION'),
    )
except Exception as e:
    st.error(f"Error creating AWS session: {e}")
    st.stop()

model_id = "anthropic.claude-3-haiku-20240307-v1:0"

model_kwargs = {
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

# ------------------------------------------------------
# LangChain - RAG chain with citation
