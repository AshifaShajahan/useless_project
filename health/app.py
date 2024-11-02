
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from transformers import pipeline

# Load and process the PDF
@st.cache_resource
def load_pdf():
    pdf_name = 'BOOK.pdf'  # Ensure this file exists in the same directory
    loaders = [PyPDFLoader(pdf_name)]

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)

    return index

# Load the PDF document
index = load_pdf()

# Initialize the text generation pipeline with a local model (e.g., GPT-2)
text_generator = pipeline("text-generation", model="gpt2")

# Set up the Streamlit app
st.title('MEDICAL BOT 🤖')

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# User input for the prompt
prompt = st.chat_input('Pass Your Prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Retrieve context from the index
    docs = index.vectorstore.as_retriever().get_relevant_documents(prompt)  # Get relevant documents

    # Combine the content of retrieved documents
    context = ' '.join([doc.page_content for doc in docs]) if docs else ""

    if context:
        # Generate a response using the local model
        full_prompt = f"{context}\n\nUser: {prompt}\nAssistant:"
        result = text_generator(full_prompt, max_length=150, num_return_sequences=1)

        # Display the response
        response = result[0]['generated_text'].split("Assistant:")[-1].strip()
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
    else:
        st.chat_message('assistant').markdown("Sorry, I couldn't find any relevant information.")
