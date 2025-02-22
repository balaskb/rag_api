from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chat_models import ChatOpenAI
from flask import Flask, render_template_string, request
import os
api_key = input("Please enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key
print("OPENAI_API_KEY has been set!")
txt_file_path = 'tourism.txt'
loader = TextLoader(file_path=txt_file_path, encoding="utf-8")
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
data = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings()
import faiss
vectorstore = FAISS.from_documents(data, embedding=embeddings)
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    memory=memory
)

def generate_tamilnadu_tourism_details(components):
  result = conversation_chain({"question": components})
  answer = result["answer"]
  return answer


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello():
  output = ""
  if request.method == 'POST':
    components = request.form['components']
    output = generate_tutorial(components)


# This is a HTML template for a Custom Recipe Generator web page. It includes a form for users to input a list of ingredients/items they have, and two JavaScript functions for generating a recipe based on the input and copying the output to the clipboard. The template uses the Bootstrap CSS framework for styling.
  return render_template_string('''

 <!DOCTYPE html >
 <html >
 <head >
  <title >Tamil Nadu Tourism Bot</title >
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css"
    rel="stylesheet">
  <script >

  async function generateTutorial() {
   const components = document.querySelector('#components').value;
   const output = document.querySelector('#output');
   output.textContent = 'Fetching the details for you...';
   const response = await fetch('/generate', {
    method: 'POST',
    body: new FormData(document.querySelector('#tutorial-form'))
   });
   const newOutput = await response.text();
   output.textContent = newOutput;
  }
  function copyToClipboard() {
   const output = document.querySelector('#output');
   const textarea = document.createElement('textarea');
   textarea.value = output.textContent;
   document.body.appendChild(textarea);
   textarea.select();
   document.execCommand('copy');
   document.body.removeChild(textarea);
   alert('Copied to clipboard');
  }

  </script >
 </head >
 <body >
  <div class="container">
   <h1 class="my-4">Tamil Nadu Travel Guide Q&A Bot</h1 >
   <form id="tutorial-form" onsubmit="event.preventDefault(); generateTutorial();" class="mb-3">
    <div class="mb-3">
     <label for="components" class="form-label">Ask Your Question:</label >
     <input type="text" class="form-control" id="components" name="components" placeholder="Enter the destination detials to get the answer e.g. Top 10 Tourist Places in Tamil Nadu" required >
    </div >
    <button type="submit" class="btn btn-primary">Get the Answer</button >
   </form >
   <div class="card">
    <div class="card-header d-flex justify-content-between align-items-center">
     Output:
     <button class="btn btn-secondary btn-sm" onclick="copyToClipboard()">Copy </button >
    </div >
    <div class="card-body">
     <pre id="output" class="mb-0" style="white-space: pre-wrap;">{{ output }}</pre >
    </div >
   </div >
  </div >
 </body >
 </html >


 ''',
                                output=output)


@app.route('/generate', methods=['POST'])
def generate():
  components = request.form['components']
  return generate_tamilnadu_tourism_details(components)


if __name__ == '_main_':
  app.run(host='0.0.0.0', port=8080)