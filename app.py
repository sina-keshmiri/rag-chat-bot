import streamlit as st
from utils import extract_pdf_text, get_text_chunks, create_vector_store, create_conversation_chain

title = "Chat with multiple PDFs"

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
                raw_text = extract_pdf_text(pdf_docs)
                st.write('text extraction: extracted')
                text_chunks = get_text_chunks(raw_text)
                st.write('text chunks: created')
                vectorstore = create_vector_store(text_chunks)
                st.write('creating and embedding data: completed')
                st.session_state.conversation = create_conversation_chain(vectorstore)
                st.write('Conversation chain creation: created')

main()