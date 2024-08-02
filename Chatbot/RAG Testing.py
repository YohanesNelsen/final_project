import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import spacy
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate, LLMChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = re.sub(r'[^a-zA-Z0-9\s.,\'-]', '', text)
    text = re.sub(' +', ' ', text).strip().lower()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words or token == 'not']
    return ' '.join(filtered_tokens)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if 'conversation' in st.session_state and callable(st.session_state.conversation):
        knowledge_level = st.session_state.knowledge_level
        
        # Define knowledge level context
        if knowledge_level == "Beginner":
            context_description = (
                "Use words that are used by elementary students. The explanation should be for someone with a low to mid income (quintile 1-3) "
                "and primary to lower secondary education, around the level of 7 and 9 grader with low literacy and numeracy scores. "
                "Keep the sentence short and use one or two-syllable words."
                "If any financial term is mentioned, also briefly explain related concepts."
                "Answer in 2 paragraph maximum"
            )
        elif knowledge_level == "Intermediate":
            context_description = (
                "The explanation should be for someone with a low to mid income (quintile 1-3) "
                "and upper secondary to post-secondary non-tertiary education (ISCED 3C - ISCED 4C). "
                "Assume medium literacy and numeracy scores. "
                "Use words with 1-3 syllables."
                "If any financial term is mentioned, also briefly explain related concepts."
                "Answer in 2 paragraph maximum"
            )
        else:  # Advanced
            context_description = (
                "The explanation should be for someone with a mid to high income (quintile 4-5) "
                "and tertiary education (ISCED 5 and above). "
                "Assume high literacy and numeracy scores. "
                "Use precise and accurate terminology, but ensure clarity."
                "If any financial term is mentioned, also briefly explain related concepts."
                "Answer in 2 paragraph maximum"
            )

        # Construct the modified question with additional context
        modified_question = (
            f"{user_question}. If this question is not related to financial/law, say 'I'm sorry, I don't have the information you need. "
            f"Also give a short example if possible. Please explain in a way suitable for a {knowledge_level} level. {context_description}"
        )
        
        # Get the response from the conversation chain
        response = st.session_state.conversation({'question': modified_question})
        
        # Add the user question and response to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
        
        # Automatically generate and ask the follow-up question
        follow_up_question = f"From where in the file can you get the statement: '{response['answer']}'?"
        follow_up_response = st.session_state.conversation({'question': follow_up_question})
        
        # Add follow-up response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": follow_up_response['answer']})
        
        # Display the follow-up response at the top
        st.write(bot_template.replace("{{MSG}}", follow_up_response['answer']), unsafe_allow_html=True)

        # Display the rest of the chat history
        for message in reversed(st.session_state.chat_history[:-1]):  
            if message["role"] == "assistant":
                st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
            else:
                st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
    else:
        st.error("Please refresh the page, submit and process a PDF file before asking questions.")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "extracted_legal_terms" not in st.session_state:
        st.session_state.extracted_legal_terms = None
    if "show_input" not in st.session_state:
        st.session_state.show_input = True  
    if "knowledge_level" not in st.session_state:
        st.session_state.knowledge_level = "Beginner"

    st.header("Chat with multiple PDFs :books:")

    if st.session_state.show_input:
        user_question = st.text_input("Ask a question about your documents:", key="user_question")
        if user_question:
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        
        knowledge_level = st.selectbox(
            "Select your knowledge level about legal documents:",
            ("Beginner", "Intermediate", "Advanced"),
            key="knowledge_level"
        )
        
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, key="pdf_uploader")
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

if __name__ == '__main__':
    main()