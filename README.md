RAG stands for Retrieval-Augmented Generation. This technique combines information retrieval with text generation to create more accurate and contextually relevant responses. In essence, RAG retrieves relevant information from a large dataset or database and then uses this information to generate coherent and factually grounded text. This approach is particularly useful in applications like question answering, summarization, and conversational agents, where providing accurate and detailed responses is crucial.

Here's a simple diagram to illustrate how the RAG process is used in a simple use-case, document context-aware question answering with an LLM:


![This is an image](https://github.com/Haste171/rag-demo/blob/main/diagram.png)  

## Converting this Diagram Into Code

**1. Imports and Setup**
First, we import the necessary libraries and load environment variables.

```python
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
```
`Here, we are importing libraries for handling PDF loading, text splitting, embedding, vector store, and environment variables.`

**2. Initialize Clients and Utilities**
We initialize the OpenAI client and other utilities required for text splitting and embeddings.

```python
client = OpenAI(api_key=OPENAI_API_KEY)  # Using OpenAI client directly
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)  # Langchain wrapper for OpenAI embeddings
persist_directory = 'database/'
docsearch = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
```
`We initialize the OpenAI client and configure the text splitter and embedding functions.`

**3. Parse PDF Function**
This function loads a PDF file, splits it into chunks, and returns the chunks as documents.

```python
def parse_pdf(pdf_path: str) -> List[Document]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found at {pdf_path}")
    
    if not pdf_path.endswith('.pdf'):
        raise ValueError(f"File at {pdf_path} is not a PDF file")
    
    loader = PyPDFLoader(pdf_path)
    return text_splitter.split_documents(loader.load())
```
`The parse_pdf function takes a PDF file path, verifies its existence and format, loads the PDF, and splits it into manageable chunks.`

**4. Ingest Data Function**
This function ingests data into a vector store, making it searchable.

```python
def ingest_data(data: List[Document], persist_directory: str = "database/") -> None:
    vectorstore = Chroma.from_documents(
        documents=data, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    vectorstore.persist()
```
`The ingest_data function takes the parsed documents and stores them in a vector store, enabling efficient similarity search.`

**5. Generate Prompt Function**
This function generates a prompt for the LLM by retrieving similar documents based on a query.

```python
def gen_prompt(query: str, k: int = 4) -> str:
    docs = docsearch.similarity_search(query, k)
    
    print(f"Similar docs: ")
    for i, doc in enumerate(docs):
        print(f"{Fore.WHITE}Doc {i+1}:")
        print(f"{Fore.BLUE}~~~")
        print(doc.page_content)
        print(f"~~~{Fore.RESET}")
    print(Fore.RESET)
    
    return f"""To answer the question please only use the Context given, nothing else. Do not make up answer, simply say 'I don't know' if you are not sure.\nQuestion: {query}\nContext: {[doc.page_content for doc in docs]}\n"""
```
`The gen_prompt function performs a similarity search to find relevant documents and constructs a prompt including the context for the question.`

**6. Stream Function**
This function streams the response from the LLM using the generated prompt.

```python
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
```
`The stream function generates the final prompt, sends it to the LLM, and streams the response back, displaying it in real-time.`

**7. Main Execution**
Finally, we parse the PDF, ingest the data, and query the LLM.

```python
data = parse_pdf("example_data/bitcoin_whitepaper.pdf")
ingest_data(data)

input_text = "What is Bitcoin?"
for response in stream(input_text):
    print(response, end="")
```
`We parse a PDF file, ingest its content into the vector store, and then query the LLM to answer a question about Bitcoin, streaming and printing the response.`

I hope you found this tutorial useful. Feel free to explore the code and adapt it to your needs.

Credits to [Dante Noguez](https://github.com/DanteNoguez) & [Mayo Oshin](https://github.com/mayooear) for inspiration
