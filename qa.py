import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#Langsimth Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "QA Chatbot with OPEN AI"

#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assitant. Please response to the user queries"),
        ("user", "Question : {question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    openai.api_key = api_key,
    llm = ChatOpenAI(model = llm)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question": question})
    return answer

#streamlit 
st.title("Q&A Chatbot with OpenAI")

#side bar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type = "password")

#drop down to select various openai models
llm = st.sidebar.selectbox("Select an OpenAI model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])

#adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0, value=0.7)
max_tokens= st.sidebar.slider("Max Tokens",min_value=50, max_value=300, value=150)

#main interface for user input
st.write("Ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Write your query")