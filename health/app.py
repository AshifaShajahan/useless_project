from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # Use transformers for conversational AI
import os

# Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Replace with your Hugging Face token

# Load and process the PDF
@st.cache_resource
def load_pdf():
    pdf_name = 'BOOK1.pdf'  # Ensure this file exists in the same directory
    loaders = [PyPDFLoader(pdf_name)]

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)

    return index

# Load the PDF document
index = load_pdf()

# Load your custom model and tokenizer from the Hugging Face Hub
model_name = "Anjanams04/healthbot"  # Replace with your Hugging Face model repository
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a conversational pipeline using your fine-tuned model
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

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

    # Greeting responses
    if any(greet in prompt.lower() for greet in ["hi", "hello", "hey"]):
        greeting_response = "Hello! How can I assist you today?"
        st.chat_message('assistant').markdown(greeting_response)
        st.session_state.messages.append({'role': 'assistant', 'content': greeting_response})

    # Thank you responses
    elif "thank you" in prompt.lower():
        thank_you_response = "You're welcome! If you have any more questions, feel free to ask."
        st.chat_message('assistant').markdown(thank_you_response)
        st.session_state.messages.append({'role': 'assistant', 'content': thank_you_response})

    # Check for symptoms and respond accordingly
    else:
        # Retrieve context from the index
        docs = index.vectorstore.as_retriever().invoke(prompt)  # Get relevant documents

        # Combine the content of retrieved documents
        context = ' '.join([doc.page_content for doc in docs]) if docs else ""

        if context:
            # Use the conversational model to generate a response with context
            full_prompt = f"{context}: {prompt}:"
            response = chatbot(full_prompt, num_return_sequences=1, max_new_tokens=1000, truncation=True)  # Use max_new_tokens

            # Display the response
            st.chat_message('assistant').markdown(response[0]['generated_text'])
            st.session_state.messages.append({'role': 'assistant', 'content': response[0]['generated_text']})
        else:
            st.chat_message('assistant').markdown("Sorry, I couldn't find any relevant information.")
