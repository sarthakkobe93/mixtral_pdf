import streamlit as st
from streamlit_lottie import st_lottie
import fitz  # PyMuPDF
import requests
import os, shutil
import llm_model


SYSTEM_PROMPT = [
    """
    You are not Mistral AI, but rather a Q&A bot trained by Krishna Kumar while building a cool side project based on RAG. Whenever asked, you need to answer as Q&A bot.
    """,
    """You are a RAG based Document Q&A bot. Based on the input prompt and retrieved context from the vector database you will answer questions that are closer to the context. 
    If no context was found then, say "I don't know" instead of making up answer on your own. Follow above rules strictly.
    """
]


@st.cache_data(experimental_allow_widgets=True)
def index_document(_llm_object, uploaded_file):

    if uploaded_file is not None:
        # Specify the folder path where you want to store the uploaded file in the 'assets' folder
        assets_folder = "assets/uploaded_files"
        if not os.path.exists(assets_folder):
            os.makedirs(assets_folder)

        # Save the uploaded file to the specified folder
        file_path = os.path.join(assets_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        file_name = os.path.join(assets_folder, uploaded_file.name)
        st.success(f"File '{file_name}' uploaded !")

        with st.spinner("Indexing document... This is a free CPU version and may take a while ⏳"):
            retriever = _llm_object.create_vector_db(file_name)
            st.session_state.retriever = retriever
            
        return file_name
    else:
        return None


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def is_query_valid(query: str) -> bool:
    if not query:
        st.error("Please enter a question!")
        return False
    return True

def init_state() :
    if "filename" not in st.session_state:
        st.session_state.filename = None
        
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "temp" not in st.session_state:
        st.session_state.temp = 0.7

    if "history" not in st.session_state:
        st.session_state.history = [SYSTEM_PROMPT]

    if "repetion_penalty" not in st.session_state :
        st.session_state.repetion_penalty = 1

    if "chat_bot" not in st.session_state :
        st.session_state.chat_bot = "Mixtral-8x7B-Instruct-v0.1"
        

def faq():
    st.markdown(
        """
        # FAQ
        ## How does Document Q&A Bot work?
        When you upload a document (in Pdf, word, csv or txt format), it will be divided into smaller chunks 
        and stored in a special type of database called a vector index 
        that allows for semantic search and retrieval.

        When you ask a question, our Q&A bot will first look through the document chunks and find the
        most relevant ones using the vector index. This acts as a context to our custom prompt which will be feed to the LLM model.
        If the context was not found in the document then, LLM will reply 'I don't know'

        ## Is my data safe?
        Yes, your data is safe. Our bot does not store your documents or
        questions. All uploaded data is deleted after you close the browser tab.

        ## Why does it take so long to index my document?
        Since, this is a sample QA bot project that uses open-source model
        and doesn't have much resource capabilities like GPU, it may take time 
        to index your document based on the size of the document.

        ## Are the answers 100% accurate?
        No, the answers are not 100% accurate. 
        But for most use cases, our QA bot is very accurate and can answer
        most questions. Always check with the sources to make sure that the answers
        are correct.
        """
    )


def sidebar():
    with st.sidebar:
        st.markdown("## Document Q&A Bot")
        st.write("LLM: Mixtral-8x7B-Instruct-v0.1")
        #st.success('API key already provided!', icon='✅')
               
        st.markdown("### Set Model Parameters")
        # select LLM model
        st.session_state.model_name = 'Mixtral-8x7B-Instruct-v0.1'
        # set model temperature
        st.session_state.temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.7)
        st.session_state.top_p = st.slider(label="Top Probablity", min_value=0.0, max_value=1.0, step=0.1, value=0.95)
        st.session_state.repetition_penalty = st.slider(label="Repetition Penalty", min_value=0.0, max_value=1.0, step=0.1, value=1.0)
        
        # load model parameters
        st.session_state.llm_object = load_model()
        st.markdown("---")
        # Upload file through Streamlit
        st.session_state.uploaded_file = st.file_uploader("Upload a file", type=["pdf", "doc", "docx", "txt"])
        index_document(st.session_state.llm_object, st.session_state.uploaded_file)
        
        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            """QA bot 🤖 allows you to ask questions about your 
            documents and get accurate answers with citations."""
        )

        st.markdown("Created with ❤️ by Krishna Kumar Yadav")
        st.markdown(
            """
            - [Github](https://github.com/sarthakkobe93)
            """
        )

        faq()


def chat_box() :
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            

def generate_chat_stream(prompt) :
    
    with st.spinner("Fetching relevant answers from source document..."):
        response, sources = st.session_state.llm_object.mixtral_chat_inference(prompt, st.session_state.history, st.session_state.temperature, 
                                                                               st.session_state.top_p, st.session_state.repetition_penalty, st.session_state.retriever)
    
        
    return response, sources

def stream_handler(chat_stream, placeholder) :
    full_response = ''

    for chunk in chat_stream :
        if chunk.token.text!='</s>' :
            full_response += chunk.token.text
            placeholder.markdown(full_response + "▌")
    placeholder.markdown(full_response)

    return full_response

def show_source(sources) :
    with st.expander("Show source") :
        for source in sources:
            st.info(f"{source}")
            

# Function to load model parameters
@st.cache_resource()
def load_model():
    # create llm object
    return llm_model.LlmModel()

st.set_page_config(page_title="Document QA Bot")
lottie_book = load_lottieurl("https://assets4.lottiefiles.com/temp/lf20_aKAfIn.json")
st_lottie(lottie_book, speed=1, height=200, key="initial")
# Place the title below the Lottie animation
st.title("Document Q&A Bot 🤖")

# initialize session state for streamlit app
init_state()
# Left Sidebar
sidebar()
chat_box()

if prompt := st.chat_input("Ask a question about your document!"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        chat_stream, sources = generate_chat_stream(prompt)
  
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = stream_handler(chat_stream, placeholder)
            show_source(sources)

        st.session_state.history.append([prompt, full_response])
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    except Exception as e:
        if not st.session_state.uploaded_file:
            st.error("Kindly provide the document file by uploading it before posing any questions. Your cooperation is appreciated!")
        else:
            st.error(e)
