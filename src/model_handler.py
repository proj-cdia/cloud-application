from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from utils import * 

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
PROMPT_TEMPLATE = """Com base no contexto e nas informações fornecidas pelo usuário, recomende produtos que atendam às suas necessidades, incluindo nome e especificações do item. Se você não souber a resposta, apenas diga que não sabe, não tente inventar uma resposta.  
Informações do usuário:
"""  
data_path = 'data'
documents = []
for file in os.listdir(data_path):
    file_path = os.path.join(data_path, file)
    if file.endswith('.pdf'):
        pass
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx'):
        pass

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = HuggingFaceHubEmbeddings()
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={'k':1})

llm = HuggingFaceHub(repo_id='google/flan-t5-xl', model_kwargs={"temperature":0.1, "max_new_tokens":250})
global qa 
qa = RetrievalQA.from_chain_type(llm=llm, 
                                 chain_type="stuff", 
                                 retriever=retriever, 
                                 return_source_documents=True)

while True:
    question = 'Em que ano foi fundada a Terra dos Robôs?'
    prompt = PROMPT_TEMPLATE + question
    print(prompt)
    # answer = qa(question)
    # process_llm_response(answer)