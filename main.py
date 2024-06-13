import streamlit as st
import time
from dotenv import load_dotenv
import os

import boto3
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import BedrockChat
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

load_dotenv()
# Set page configuration
st.set_page_config(page_title='Knowledge Bases for Amazon Bedrock and LangChain 🦜️🔗')

# ------------------------------------------------------
# Amazon Bedrock - settings
try:
    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE_NAME'))
    bedrock_runtime = session.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
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
# LangChain - RAG chain with citations
template = '''Answer the question based only on the following context:
{context}

Question: {question}'''

prompt_template = ChatPromptTemplate.from_template(template)

# Amazon Bedrock - KnowledgeBase Retriever
try:
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=os.getenv('BEDROCK_KNOWLEDGE_BASE_ID'),
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},  # Retrieve multiple passages
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

st.title("Green Mountain Girls Farm Instruction Manual Assistant")

# Clear Chat History function
def clear_screen():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

with st.sidebar:
    st.title('Knowledge Bases for Amazon Bedrock and LangChain 🦜️🔗')
    streaming_on = st.checkbox('Streaming')
    st.button('Clear Screen', on_click=clear_screen)

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Selection of tractor model
tractor_models = ["Woodchipper", "Zero Turn Mower", "Steam Kettle Scalder"]  # Replace with actual models
selected_model = st.selectbox("Select the piece of equipment you need help with:", tractor_models)

# Chat Input - User Prompt
if user_prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    query_context = f"Tractor Model: {selected_model}\n{user_prompt}"

    if streaming_on:
        # Chain - Stream
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ''
            full_context = []
            # Chain Stream
            for chunk in chain.stream(query_context):
                if 'response' in chunk:
                    full_response += chunk['response']
                    placeholder.markdown(full_response)
                elif 'context' in chunk:
                    full_context.extend(chunk['context'])

            # Check if full_context is non-empty
            if full_context:
                # Filter out non-Document objects
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
        # Chain - Invoke
        with st.chat_message("assistant"):
            response = chain.invoke(query_context)

            # Check if response['context'] is non-empty
            if response['context']:
                # Filter out non-Document objects
                response['context'] = [doc for doc in response['context'] if hasattr(doc, 'metadata')]

                if response['context']:
                    # Identify the top document based on confidence score
                    top_document = max(response['context'], key=lambda x: x.metadata.get('score', 0))
                    top_document_uri = top_document.metadata['source_metadata']['x-amz-bedrock-kb-source-uri']

                    # Filter passages to only include those from the top document
                    filtered_context = [ctx for ctx in response['context'] if ctx.metadata['source_metadata']['x-amz-bedrock-kb-source-uri'] == top_document_uri]

                    st.write(response['response'])
                    with st.expander("Show source details >"):
                        st.write(filtered_context)
                    st.session_state.messages.append({"role": "assistant", "content": response['response']})
                else:
                    st.error("No valid context retrieved.")
            else:
                st.error("No context retrieved.")
