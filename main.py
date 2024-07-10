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


load_dotenv()


os.environ['AWS_REGION'] = 'us-east-1'


try:
    botocore_session = botocore.session.get_session()
    client = botocore_session.create_client('secretsmanager', region_name=os.getenv('AWS_REGION'))
    cache_config = SecretCacheConfig()
    cache = SecretCache(config=cache_config, client=client)
    secret = json.loads(cache.get_secret_string('gmgf_secrets'))
except Exception as e:
    st.error(f"Error accessing Secrets Manager: {e}")
    st.stop()


st.set_page_config(page_title='Green Mountain Girls Farm Instruction Manual Assistant')


authenticator = CognitoAuthenticator(
    pool_id=secret['POOL_ID'],
    app_client_id=secret['CLIENT_ID'],
    app_client_secret=secret['CLIENT_SECRET'],
    use_cookies=True
)


is_logged_in = authenticator.login()
if not is_logged_in:
    st.stop()

def logout():
    authenticator.logout()


try:
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


template = '''Answer the question based only on the following context:
{context}

Question: {question}'''

prompt_template = ChatPromptTemplate.from_template(template)


try:
    credentials = session.get_credentials()
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=secret['BEDROCK_KNOWLEDGE_BASE_ID'],
        region_name=os.getenv('AWS_REGION'),
        aws_access_key_id=credentials.access_key,
        aws_secret_access_key=credentials.secret_key,
        aws_session_token=credentials.token,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
    )
except Exception as e:
    st.error(f"Error creating AmazonKnowledgeBasesRetriever: {e}")
    st.stop()

model = BedrockChat(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    .assign(response=prompt_template | model | StrOutputParser())
    .pick(["response", "context"])
)

def clear_screen():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

with st.sidebar:
    st.title('Green Mountain Girls Farm Instruction Manual Assistant')
    
    streaming_on = st.checkbox('Streaming')
    st.button('Clear Screen', on_click=clear_screen)
    st.divider()
    st.button("Logout", "logout_btn", on_click=logout)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


tractor_models = ["Woodchipper", "Zero Turn Mower", "Steam Kettle Scalder", "Antonio Carraro serie 30", "Freeaire Walk-In", "Salad Dryer", "Gator", "Antonio Carraro TRX 7800 S", "Turbo air refrigerator"]  # Replace with actual models
selected_model = st.selectbox("Select the piece of equipment you need help with:", tractor_models)


if user_prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    query_context = f"Tractor Model: {selected_model}\n{user_prompt}"

    if streaming_on:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ''
            full_context = []
            for chunk in chain.stream(query_context):
                if 'response' in chunk:
                    full_response += chunk['response']
                    placeholder.markdown(full_response)
                elif 'context' in chunk:
                    full_context.extend(chunk['context'])

            if full_context:
                full_context = [doc for doc in full_context if hasattr(doc, 'metadata')]

                if full_context:
                    # Identify the top document based on confidence score
                    top_document = max(full_context, key=lambda x: x.metadata.get('score', 0))
                    top_document_uri = top_document.metadata['source_metadata']['x-amz-bedrock-kb-source-uri']

                    # Filter passages to only include those from the top document
                    filtered_context = [ctx for ctx in full_context if ctx.metadata['source_metadata']['x-amz-bedrock-kb-source-uri'] == top_document_uri]

                    with st.expander("Show source details >"):
                        st.write(filtered_context)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("No valid context retrieved.")
            else:
                st.error("No context retrieved.")
    else:
        with st.chat_message("assistant"):
            response = chain.invoke(query_context)

            if response['context']:
                response['context'] = [doc for doc in response['context'] if hasattr(doc, 'metadata')]

                if response['context']:
                    top_document = max(response['context'], key=lambda x: x.metadata.get('score', 0))
                    top_document_uri = top_document.metadata['source_metadata']['x-amz-bedrock-kb-source-uri']

                    filtered_context = [ctx for ctx in response['context'] if ctx.metadata['source_metadata']['x-amz-bedrock-kb-source-uri'] == top_document_uri]

                    st.write(response['response'])
                    with st.expander("Show source details >"):
                        st.write(filtered_context)
                    st.session_state.messages.append({"role": "assistant", "content": response['response']})
                else:
                    st.error("No valid context retrieved.")
            else:
                st.error("No context retrieved.")
