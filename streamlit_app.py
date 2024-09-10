import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import chromadb
from chromadb.config import Settings
from uuid import uuid4
from langchain_core.documents import Document
from chromadb import Client
from langchain_community.embeddings import GPT4AllEmbeddings
import ollama
import torch
import ast


client = Client(Settings())
vectorstore = Chroma(
    collection_name="texts",
    embedding_function=GPT4AllEmbeddings(),
    persist_directory="./chroma_langchain_db",
)
llm = OllamaLLM(model="llama3")
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""
def query_rag(query_text, vectorstore):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
    Args:
      - query_text (str): The text to query the RAG system with.
    Returns:
      - formatted_response (str): Formatted response including the generated text and sources.
      - response_text (str): The generated response text.
    """

    # Retrieving the context from the DB using similarity search
    results = vectorstore.similarity_search_with_relevance_scores(query_text, k=1)


    # Check if there are any matching results or if the relevance score is too low
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find close matching results.")

    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])


    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)


    # Generate response text based on the prompt
    response_text = llm.predict(prompt)

    # Get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _score in results]

    # Format and return response including generated text and sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text
st.title("Question Answering with Llama3")

question = st.text_input("Enter your question:")
if question:
    formatted_response, response_text = query_rag(question, vectorstore)
    st.write("Response:")
    response_placeholder = st.empty()
    # Append each piece of the response to the output

    final_response = response_text
    response_placeholder.write(final_response)


    #st.write(final_response)
