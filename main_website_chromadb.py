from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import os
from langchain.document_loaders import UnstructuredURLLoader
from langchain_chroma import Chroma
import requests
from bs4 import BeautifulSoup
import streamlit as st

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def fetch_sitemap(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses
    return response.content

def parse_sitemap(xml_content):
    soup = BeautifulSoup(xml_content, 'lxml-xml')  # Use 'lxml' parser
    sitemap_urls = [loc.text for loc in soup.find_all('loc')]
    return sitemap_urls

def get_all_urls(sitemap_url):
    all_urls = []
    urls_to_process = [sitemap_url]

    while urls_to_process:
        current_url = urls_to_process.pop()
        content = fetch_sitemap(current_url)
        urls = parse_sitemap(content)

        for url in urls:
            if url.endswith('.xml'):
                urls_to_process.append(url)
            else:
                all_urls.append(url)

    return all_urls
    
class ChatBot():
    load_dotenv()
    # URL_L=['https://urbanemissions.info/','https://urbanemissions.info/blog-pieces/whats-polluting-delhis-air/','https://urbanemissions.info/india-apna/'
    #         'https://urbanemissions.info/tools/','https://urbanemissions.info/tools/vapis/','https://urbanemissions.info/tools/aqi-calculator','https://urbanemissions.info/tools/atmos',
    #         'https://urbanemissions.info/tools/wrf-configuration-notes/','https://urbanemissions.info/india-air-quality/india-ncap-review/#sim-40-2021',
    #         'https://urbanemissions.info/blog-pieces/airquality-smog-towers-crazy/','https://urbanemissions.info/blog-pieces/delhi-diwali-2017-understanding-emission-loads/',
    #         'https://urbanemissions.info/blog-pieces/resources-how-to-access-aqdata-in-india/','https://urbanemissions.info/india-emissions-inventory/emissions-in-india-open-agricultural-forest-fires/']
    initial_sitemap_url = 'https://urbanemissions.info/sitemap.xml'
    all_extracted_urls = get_all_urls(initial_sitemap_url)
    loader = UnstructuredURLLoader(urls=all_extracted_urls)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    # pinecone_api_key = os.getenv('PINECONE_API_KEY')
    # pinecone = PineconeClient(api_key=pinecone_api_key)

    # index_name = "langchain-demo-2"

    # if index_name in pinecone.list_indexes().names():  
    #     pinecone.delete_index(index_name)  

    # if index_name not in pinecone.list_indexes().names():
    #     pinecone.create_index(
    #         name=index_name, 
    #         dimension=768, 
    #         metric='cosine', 
    #         spec=ServerlessSpec(
    #             cloud='aws', 
    #             region='us-east-1'
    #         )
    #     )
    #     docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    # # else:
    # #     docsearch = Pinecone.from_existing_index(index_name, embeddings)
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs={"temperature": 0.8, "top_k": 50}, huggingfacehub_api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
    
    )

    from langchain import PromptTemplate


    import bs4
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.prompts import ChatPromptTemplate
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

    from langchain_core.runnables import RunnableParallel

    system_prompt = (
    "You are an assistant for question-answering on a data related to enviornment. "
    "Use the following pieces of retrieved-context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use 5 sentences minimum and keep the "
    "answer elaborate."
    "\n\n"
    "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough


    # retreiver=docsearch.as_retriever()


    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    retrieve_docs = (lambda x: x["input"]) | retriever

    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )

    # rag_chain_from_docs = (
    #     RunnablePassthrough.assign(context=(lambda x: format_hell(x["context"])))
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # rag_chain_with_source = RunnableParallel(
    #     {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
    # ).assign(answer=rag_chain_from_docs)



# bot =ChatBot()

# result = bot.chain.invoke({"input": "Tell me about vapis"})
# print(result)

