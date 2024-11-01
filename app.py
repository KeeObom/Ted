# Libraries
import streamlit as st
from chat_documents import DocumentChatBot
from chat_sql import SQLChatBot

# Initialize chatbots
doc_bot = DocumentChatBot()
sql_bot = SQLChatBot()

st.title("Document and SQL Chatbot")


page = st.sidebar.selectbox("Choose Page", ["Chat with Documents", "Chat with SQL Database"])

if page == "Chat with Documents":
    st.header("Chat with Your Documents")
    user_query = st.text_input("Ask something about the document:")

    if st.button("Submit"):
        response = doc_bot.answer_query(user_query)
        st.write("Bot:", response)

elif page == "Chat with SQL Database":
    st.header("Chat with SQL Database")
    user_query = st.text_input("Ask something about the SQL data:")

    if st.button("Submit"):
        response = sql_bot.answer_query(user_query)
        st.write("Bot:", response)
