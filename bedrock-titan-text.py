from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st
from langchain.globals import set_verbose

set_verbose(True)

os.environ["AWS_PROFILE"] = "andre"

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"
)

modelID = "amazon.titan-text-lite-v1"

llm = BedrockLLM(
    model_id=modelID,
    client=bedrock_client,
)

def chatbot(language, question):
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a knowledgeable assistant. Answer questions based on the information you have. Language: {language}. Question: {freeform_text}. Answer:"
    )
    
    prompt_text = prompt.format(language=language, freeform_text=question)
    
    try:
        response = llm.invoke(prompt_text)
        if response:
            return response
        else:
            return "Sorry, I couldn't get a valid response from the model."
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

st.title("Bedrock Chatbot")
language = st.sidebar.selectbox("Language", ["english", "portuguese"])

if language:
    freeform_text = st.sidebar.text_area(label="What is your question?", max_chars=100)
    submit_button = st.sidebar.button("Submit")

    if submit_button and freeform_text:
        with st.spinner('Thinking...'):
            response = chatbot(language, freeform_text)
            st.write(response)
    elif submit_button:
        st.write("Please enter a question.")
