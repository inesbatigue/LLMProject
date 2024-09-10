import streamlit as st
from streamlit_chat import message
import re
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
import base64


def validate_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# Fake login function (accepts any username and password)
def login(username, password):
    return True  # Always returns True

# Dictionary to store created accounts
accounts = {}

# Utility to add background image and logo
def add_background_and_logo(bg_image, logo_image):
    # Background Image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({bg_image});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Logo
    st.markdown(
        f"""
        <div style='text-align:center; padding: 10px;'>
            <img src="{logo_image}" width="150" height="150">
        </div>
        """,
        unsafe_allow_html=True
    )@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set background image
def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Display logo using st.image()
def display_logo(logo_file):
    st.image(logo_file, width=150)

def set_sidebar_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    sidebar_bg_img = '''
    <style>
    .sidebar .sidebar-content {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    
}
    </style>
    ''' % bin_str
    st.sidebar.markdown(sidebar_bg_img, unsafe_allow_html=True)


# Page for account creation
def create_account_page():
    set_png_as_page_bg('OLO2DC0.jpg')  # Set background image
    #display_logo('logo.jpeg')  # Display logo
    st.logo('logo.jpeg', icon_image='logo.jpeg')
    st.title("Create Account")

    # User inputs for account creation
    first_name = st.text_input("First Name")
    last_name = st.text_input("Family Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    # Button for account creation
    if st.button("Create Account"):
        if not first_name or not last_name or not email or not password or not confirm_password:
            st.error("All fields are required!")
        elif not validate_email(email):
            st.error("Please enter a valid email address.")
        elif password != confirm_password:
            st.error("Passwords do not match!")
        else:
            # Save the account details (simple simulation)
            accounts[email] = {"first_name": first_name, "last_name": last_name, "password": password}
            st.success(f"Account created successfully for {first_name} {last_name}. You can now log in.")
            st.session_state.create_account = False  # Switch to login page after account creation
            st.session_state.logged_in = True
            st.experimental_user

# Login page function
def login_page():
    set_png_as_page_bg('OLO2DC0.jpg')  # Set background image
    #display_logo('logo.jpeg')  # Display logo
    st.logo('logo.jpeg', icon_image='logo.jpeg')
    st.title("Create Account")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

            st.session_state.logged_in = True
            st.experimental_user  # Reload the page to move to the model page


# Retrieval-Augmented Generation (RAG) system function
def query_rag(query_text, vectorstore, model):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and an LLM.
    Args:
      - query_text (str): The text to query the RAG system with.
      - model (str): The selected model (Llama 2 or Llama 3)
    Returns:
      - response_text (str): The generated response text.
    """
    results = vectorstore.similarity_search_with_relevance_scores(query_text, k=1)

    # Check if there are any matching results or if the relevance score is too low
    if len(results) == 0 or results[0][1] < 0.7:
        return "No relevant information found."

    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])

    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    {context}
     - -
    Answer the question based on the above context: {question}
    """)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Perform the model inference
    llm = OllamaLLM(model=model)  # Using selected model
    response_text = llm(prompt)

    return response_text


def query_rag2(query_text, vectorstore, model):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
    Args:
      - query_text (str): The text to query the RAG system with.
    Returns:
      - formatted_response (str): Formatted response including the generated text and sources.
      - response_text (str): The generated response text.
    """
    PROMPT_TEMPLATE = """
        Answer the question based only on the following context:
        {context}
         - -
        Answer the question based on the above context: {question}
        """
    # Retrieving the context from the DB using similarity search
    results = vectorstore.similarity_search_with_relevance_scores(query_text, k=1)


    # Check if there are any matching results or if the relevance score is too low
    if len(results) == 0 or results[0][1] < 0.4:
        print(f"Unable to find close matching results.")

    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])


    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)


    # Generate response text based on the prompt
    response_text = model.predict(prompt)

    # Get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _score in results]

    # Format and return response including generated text and sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text

# Model page interface with chat-like interface
def model_page():
    #set_png_as_page_bg('OLO2DC0.jpg')  # Set background image
    #display_logo('logo.jpeg')  # Display logo
    #set_sidebar_bg('OLO2DC0.jpg')
    st.logo('logo.jpeg', icon_image='logo.jpeg')
    #set_sidebar_bg('OLO2DC0.jpg')

    st.title("Nursery Bot Assistant")

    # Sidebar - let user choose model and let user clear the current conversation
    st.sidebar.title("Model Choice ")
    #set_sidebar_bg('OLO2DC0.jpg')
    model_name = st.sidebar.radio("Choose a model:", ("Llama 2", "Llama 3"))
    clear_button = st.sidebar.button("Clear Conversation", key="clear")

    # Map model names to their identifiers
    if model_name == "Llama 2":
        model = "llama2"  # Replace with actual Llama 2 model identifier
    else:
        model = "llama3"  # Replace with actual Llama 3 model identifier

    # reset everything
    if clear_button:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.session_state['model_name'] = []

    # Initialize the vectorstore (this should be prebuilt and saved somewhere)
    vectorstore = Chroma(
        collection_name="texts",
        embedding_function=GPT4AllEmbeddings(),
        persist_directory="./chroma_langchain_db",
    )
    llm = OllamaLLM(model=model)



    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []

    # Container for chat history
    response_container = st.container()
    # Container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            formatted_response, response_text = query_rag2(user_input, vectorstore, llm)
            print('output model:   ', formatted_response)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(formatted_response)
            st.session_state['model_name'].append(model)

    #if st.session_state['generated']:
        #with response_container:
            #for i in range(len(st.session_state['generated'])):
            # message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                #message(st.session_state["generated"][i], key=str(i))
    # st.write(f"Model used: {st.session_state['model_name'][i]}")'''

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                # Style the generated message with a blue background

                message(st.session_state["generated"][i], key=str(i))

                #st.markdown(f'<div class="message" style="text-align: left;">{st.session_state["past"][i]}</div>',
                            #unsafe_allow_html=True)
                # Display generated message with no background
            #"st.markdown(f'<div class="message" style="text-align: left;">{st.session_state["generated"][i]}</div>',
                            #unsafe_allow_html=True)

                st.write(f"Model used: {st.session_state['model_name'][i]}")


# Main app logic
def main():
    # Initialize session states
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "create_account" not in st.session_state:
        st.session_state.create_account = False

    # Account creation page
    if st.session_state.create_account:
       create_account_page()

    # If not logged in and not on account creation page, show login page
    elif not st.session_state.logged_in:
        st.title("Welcome to the RAG Model App")

        if st.button("Create Account"):
            st.session_state.create_account = True
            st.experimental_user

        login_page()

    # If logged in, show the model interface
    else:
        model_page()

if __name__ == "__main__":
    main()


