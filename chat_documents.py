# Import Libraries
import os
# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PDFMinerLoader, Docx2txtLoader
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class DocumentChatBot:
    def __init__(self):
        # Load the language model
        self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
        self.model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b")
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)

        # Load documents and create a vector store
        self.vector_store = self.load_documents()

    def load_documents(self):
        # Specify the loaders for each file type
        loader = DirectoryLoader(
            "/documents/",
            glob="*.*",
            loader_cls={
                "txt": TextLoader,
                "pdf": PDFMinerLoader,
                "docx": Docx2txtLoader,
            }
        )
        documents = loader.load()
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma.from_documents(documents, embeddings)
    
    def answer_query(self, query):
        docs = self.vector_store.similarity_search(query, k=3)
        combined_docs = " ".join([doc.page_content for doc in docs])
        return self.llm(combined_docs + query)