import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import Tool, initialize_agent
from langchain_community.utilities import WikipediaAPIWrapper
import requests
from langchain_community.tools import DuckDuckGoSearchRun

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
duckduckgo_search = DuckDuckGoSearchRun()
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)

# Define the system template for answering questions
system_template = """Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer."""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]

prompt = ChatPromptTemplate.from_messages(messages)

def main():
    st.set_page_config(page_title="Chat With Website", layout="wide", page_icon="💬")

    st.title('🦜🔗 Chat With Website')
    st.markdown("""
    This chatbot extracts text from a specified website in real time and answers questions about the content provided.
    You can ask questions related to the website content and get accurate responses based on the extracted data.
    For example, you might ask questions like ***"What is the main topic of this page?"*** or,
    ***"Can you summarize the key points?"***.
    The project repository can be found [on my Github](https://github.com/muhammad-ahsan12/MakTek-internship-Task.git).
    """)
    st.sidebar.title('🔗 Input your website URL')
    st.sidebar.write('***Ask questions below, and receive answers directly from the website.***')

    url = st.sidebar.text_input("Insert the website URL")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.chat_input("Ask a question (query/prompt)")

    if user_question and url:
        os.environ['GOOGLE_API_KEY'] = "AIzaSyA0S7F21ExbBnR06YXkEi7aj94nWP5kJho"

        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')

        text = soup.get_text(separator='\n')

        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=512, chunk_overlap=100)
        docs = text_splitter.split_text(text)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        vectordb = FAISS.from_texts(texts=docs, embedding=embeddings)
        retriever = vectordb.as_retriever()

        # Create a custom Tool for the retriever
        retriever_tool = Tool(
            name="WebsiteRetriever",
            func=retriever.get_relevant_documents,
            description="Retrieves relevant documents from the website based on the query."
        )
        
        tools = [retriever_tool, duckduckgo_search]
        
        # Ensure the prompt is compatible with the tools
        agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)
        response = agent({"input": user_question})
        answer = response['output']
                
        st.session_state.chat_history.append({"question": user_question, "answer": answer})
    
    # New section for displaying chat history
    if st.session_state.chat_history:
        for entry in st.session_state.chat_history:
            _, user_col = st.columns([1, 1])
            with user_col:
                st.markdown(f"😃 **You:**")
                st.markdown(f"<div style='background-color:#fb00ff; padding: 10px; border-radius: 10px;'>{entry['question']}</div>", unsafe_allow_html=True)
            
            bot_col, _ = st.columns([2, 1])
            with bot_col:
                st.markdown(f"🤖 **Bot:**")
                st.markdown(f"<div style='background-color:#fb00ff; padding: 10px; border-radius: 10px;'>{entry['answer']}</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()