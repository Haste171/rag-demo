import os
from typing import List, Iterator
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from colorama import Fore

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) # Using OpenAI client directly
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) # Langchain wrapper for OpenAI embeddings
persist_directory = 'database/'
docsearch = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

def parse_pdf(pdf_path: str) -> List[Document]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found at {pdf_path}")
    
    if not pdf_path.endswith('.pdf'):
        raise ValueError(f"File at {pdf_path} is not a PDF file")
    
    loader = PyPDFLoader(pdf_path)
    return text_splitter.split_documents(loader.load())

def ingest_data(data: List[Document], persist_directory: str = "database/") -> None:
    vectorstore = Chroma.from_documents(
        documents=data, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    vectorstore.persist()
    
def gen_prompt(query: str, k: int = 4) -> str:
    docs = docsearch.similarity_search(query, k)
    
    print(f"Similar docs: ")
    for i, doc in enumerate(docs):
        print(f"{Fore.WHITE}Doc {i+1}:")
        print(f"{Fore.BLUE}```")
        print(doc.page_content)
        print(f"```{Fore.RESET}")
    print(Fore.RESET)
    
    return f"""To answer the question please only use the Context given, nothing else. Do not make up answer, simply say 'I don't know' if you are not sure.\nQuestion: {query}\nContext: {[doc.page_content for doc in docs]}\n"""

def stream(input_text: str) -> Iterator[str]:
    final_prompt = gen_prompt(input_text)
    
    data = [
            {"role": "user", "content": f"{final_prompt}"},
    ]
    
    stream = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=data,
        stream=True,
    )
    
    print(f"{Fore.WHITE}Final Answer: {Fore.GREEN}")
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

data = parse_pdf("example_data/bitcoin_whitepaper.pdf")
ingest_data(data)

input_text = "What is Bitcoin?"
for response in stream(input_text):
    print(response, end="")

