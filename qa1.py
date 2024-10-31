from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import ollama
import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()

#Langsimth Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "QA Chatbot with OLLAMA"

#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assitant. Please response to the user queries"),
        ("user", "Question : {question}")
    ]
)

def generate_response(question,engine, temperature, max_tokens):

    llm = ollama(model = engine)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question": question})
    return answer


#drop down to select various openai models
llm = st.sidebar.selectbox("Select a model", ["gemma2:2b"])

#adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0, value=0.7)
max_tokens= st.sidebar.slider("Max Tokens",min_value=50, max_value=300, value=150)

#main interface for user input
st.write("Ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input,engine, temperature,max_tokens)   # check the problem
    st.write(response)
else:
    st.write("Write your query")