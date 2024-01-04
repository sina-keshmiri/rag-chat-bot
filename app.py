import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.llms import Ollama

title = "Chat with multiple PDFs"

def extract_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def create_conversation_chain(vectorstore):
    llm = Ollama(model="llama2")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_input):
    response = st.session_state.conversation({ 'question': user_input })
    st.write(response["answer"])


def main():
    st.set_page_config(page_title=title)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header(title)
    user_input = st.text_input("Ask question about your documents:")

    if user_input:
        handle_user_input(user_input)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # extract texts
                raw_text = extract_pdf_text(pdf_docs)
                st.write('text extraction: extracted')
                text_chunks = get_text_chunks(raw_text)
                st.write('text chunks: created')
                vectorstore = create_vector_store(text_chunks)
                st.write('creating and embedding data: completed')
                st.session_state.conversation = create_conversation_chain(vectorstore)
                st.write('Conversation chain creation: created')

main()