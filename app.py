from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv 


app = Flask(__name__, static_url_path='/static')

load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")

question_count = 0

max_questions = 2

static_answer_customer = '''Welcome to Zatara! 📱

Discover the power of 'Eat, Drink & Save Money' with Zatara. 🍽️

🎥 Watch our introductory video on YouTube to learn more about how Zatara connects you with exclusive offers and savings at your favorite restaurants and businesses.

📺 <a href='https://youtu.be/rlEaT_dJJrE' target='_blank'>Watch Video</a>

Ready to start 'Eating, Drinking & Saving Money' with Zatara? Explore the possibilities today!

Have questions or need assistance? Contact our support team at support@zataraapp.com. We're here to help!'''

static_answer_business = "Welcome to Zatara! 📱 Discover the power of 'Eat, Drink & Save Money' with Zatara. 🍽️ 🎥 Watch our introductory video on YouTube to learn more about how Zatara connects you with exclusive offers and savings at your favorite restaurants and businesses. 📺 <a href='https://youtu.be/DPFsjv2dFRQ' target='_blank'>Watch Video</a> Ready to start 'Eating, Drinking & Saving Money' with Zatara? Explore the possibilities today! Have questions or need assistance? Contact our support team at support@zataraapp.com. We're here to help!"

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
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

@app.route('/')
def home():
    return "Hello world"

@app.route('/customer')
def cust():
    return render_template('index.html')

@app.route('/business')
def buss():
    return render_template('base.html')

@app.route('/ask_customer', methods=['POSt','GET'])
def question():
    global question_count
    
    pdf_docs = ['customer.pdf']
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)

    if request.method == 'POST':
        user_question = request.form['question']
        print(user_question)
        
        response_text = ""

        if question_count >= max_questions:
            # Provide the static answer and reset the question count
            response_text = static_answer_customer
            question_count = 0
        else:
            pdf_docs = ['customer.pdf']
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            conversation_chain = get_conversation_chain(vectorstore)

            response = conversation_chain({'question': user_question})
            response_text = response['answer']

            question_count += 1

        print("pdf response >>>>>", response_text)

        return render_template("index.html", response_text= response_text)

    return jsonify({"error": "No file uploaded"})


@app.route('/ask_business', methods=['POSt','GET'])
def answer():
    global question_count

    if request.method == 'POST':
        user_question = request.form['question']
        print(user_question)
        
        response_text = ""

        if question_count >= max_questions:

            response_text = static_answer_business
            question_count = 0
        else:

            pdf_docs = ['business.pdf']
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            conversation_chain = get_conversation_chain(vectorstore)

            response = conversation_chain({'question': user_question})
            response_text = response['answer']

            question_count += 1

        print("pdf response >>>>>", response_text)
        
        return render_template("base.html", response_text= response_text)

    return jsonify({"error": "No file uploaded"})

if __name__ == '__main__':
    app.run(debug=True)