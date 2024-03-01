import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

# set this key as an environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets['huggingface_token']

def get_pdf_text(pdf_docs : list) -> str:
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text:str) ->list:
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1500, chunk_overlap=300, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks : list) -> FAISS:
    model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    encode_kwargs = {
        "normalize_embeddings": True
    }  # set True to compute cosine similarity
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model, encode_kwargs=encode_kwargs, model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore:FAISS) -> ConversationalRetrievalChain:
    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        #repo_id="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
        model_kwargs={"temperature": 0.5, "max_length": 1048},
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question:str):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write("   Usuario: " + message.content)
        else:
            st.write("🤖 ChatBot: " + message.content)


def main():
    st.set_page_config(
        page_title="Chat with a Bot that tries to answer questions about multiple PDFs",
        page_icon=":books:",
    )

    st.markdown("# Chat with a Bot")
    st.markdown("This bot tries to answer questions about multiple PDFs. Let the processing of the PDF finish before adding your question. 🙏🏾")

    st.write(css, unsafe_allow_html=True)

    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    
    st.header("Chat with a Bot 🤖🦾 that tries to answer questions about multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
