import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API Key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is loaded properly
if openai_api_key is None:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# Set the OpenAI API Key in the environment for OpenAI to use
os.environ["OPENAI_API_KEY"] = openai_api_key

# Define the conversation memory (to remember previous chat history)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Initialize the OpenAI chat model with the API key and use gpt-3.5-turbo
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Create a conversation chain using LangChain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Streamlit app interface
st.title("Chatbot Application")

st.write("Hello! I'm a chatbot powered by OpenAI. How can I assist you today?")

# Initialize the session state to store the chat history if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Function to process user input and display chatbot response
def chatbot_response(user_input):
    # Make an API call only when there is user input
    response = conversation.predict(input=user_input)
    return response

# User input box for chat
user_input = st.text_input("You: ", "")

# When the user submits a message, process the input and display the response
if user_input:
    # Store the user's message in session state
    st.session_state['messages'].append(f"You: {user_input}")

    # Get the chatbot's response from LangChain
    bot_response = chatbot_response(user_input)

    # Store the chatbot's response in session state
    st.session_state['messages'].append(f"Chatbot: {bot_response}")

# Display the chat history
for message in st.session_state['messages']:
    st.write(message)

# Streamlit will automatically rerun the script when the user inputs something new
