from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
from langchain.schema import Document
import requests

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def extract_html(url):

    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    for unwanted_tag in soup.find_all(['footer', 'aside', 'nav', 'script', 'style']):
        unwanted_tag.decompose()

    clean_text = soup.get_text(separator=" ", strip=True)
    clean_text = re.sub(r'\s+', ' ', clean_text)

    doc = Document(
            page_content=clean_text,
        )
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    split_docs = splitter.split_documents([doc])
    length_of_docs = len(split_docs)
    print(f"Length of docs: {length_of_docs}")
    vector = FAISS.from_documents(split_docs, embedding=embedding)
    print("Converting extracted data to vector")
    retriever = vector.as_retriever(search_kwargs={"k": 10})
    return retriever

from typing import TypedDict

class AgentState(TypedDict):
    question: str
    result: str


def is_url(text: str) -> bool:
    first_char = text.startswith("https://")
    no_space = " " not in text
    special_char = "." in text

    if first_char and no_space and special_char:
        return True
    return False

def is_available(text: str) -> bool:
    return requests.get(text).status_code == 200


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

def action(state: AgentState):
    question = state["question"]
    if is_url(question):
        if is_available:
            retriever = extract_html(question)
            print("Converting documents to retriever")
            info = retriever.invoke("sensitive terms and conditions")
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            system = f"You are an assistant who advises user on terms and conditions, user will give some extracts from the webpage. Give a detailed report on the terms and conditions outlining which terms they should take careful note of. Rank them from extremely sensitive to least. Use simple, understandable English. If the document extracts have are not terms of services or similar simply return 'Not Relevant'. Always remind that your response is simply a breakdown and should not substitute legal advice in bold capital letters at the beginning."
            prompt = ChatPromptTemplate([
                ("system", system),
                ("human", f"Document extracts:\n{"\n\n".join([i.page_content for i in info])}")
            ])

            chain = prompt | llm
            print("Using llm to get result")
            try:
                state["result"] = chain.invoke({})
                return state
            except KeyError:
                state["result"] = AIMessage(content="This website is not supported at the moment..🫤")
                return state
        else:
            state["result"] = AIMessage(content="The website is unavailable! 🤷‍♂️")
            return state
    else:
        state["result"] = AIMessage(content="Please enter a valid URL! 🙄")
        return state

from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)

graph.add_node("action", action)
graph.add_edge(START, "action")
graph.add_edge("action", END)

app = graph.compile()

import streamlit as st

st.title("Fine Print Analysis Bot 🔍")

with st.form("my_form"):
    text = st.text_input(
        label="Enter URL",
        placeholder="https://www.example.com",
    )
    submitted = st.form_submit_button("Submit Link")
    if submitted:
        st.info(app.invoke({"question": text})["result"].content)