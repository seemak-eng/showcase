import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import utils

load_dotenv()

# app config
st.set_page_config(page_title="AI Chatbot with LLM", page_icon="🤖")
st.title("Conversational Catalyst")
def get_response(user_query, chat_history):
    """
    Gets a response based on the user's query and chat history.

    Parameters:
    - user_query (str): User's query.
    - chat_history (str): History of the conversation.

    Returns:
    - str: Generated response based on the query and chat history.
    """
    # First, try to get response from Pinecone database
    query_with_contexts = utils.build_prompt(user_query)
    database_response = utils.generate_response(query_with_contexts)

    if database_response:
        print("Answer from pinecone db..")
        return database_response
    else:
        print("Answer from OpenAI..")

        # If no response from Pinecone database, chat with ChatOpenAI
        template = """
        You are a helpful assistant. Answer the following questions considering the history of the conversation:

        Chat history: {chat_history}

        User question: {user_question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatOpenAI()
            
        chain = prompt | llm | StrOutputParser()
        
        return chain.invoke({
            "chat_history": chat_history,
            "user_question": user_query,
        })
    
## For Future Enhancements
# def get_response(user_query):
#     # First, try to get response from Pinecone database
#     query_with_contexts = utils.build_prompt(user_query)
#     database_response = utils.generate_response(query_with_contexts)

#     if database_response:
#         print("Answer from Pinecone database..")
#         return database_response
#     else:
#         print("No answer found from Pinecone database..")
#         return "I'm sorry, but I don't have an answer at the moment. Please contact our support desk or check out our website for further assistance."


# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a Taylor. I am a bot. I am delighted to handle you request today!"),
    ]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Enter your query here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history)
        st.write(response)

    st.session_state.chat_history.append(AIMessage(content=response))
