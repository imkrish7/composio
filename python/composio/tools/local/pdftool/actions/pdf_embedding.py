from ast import Mod
import hashlib
from typing import Any, Dict, Literal, Optional
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import BaseModel, Field, model_validator
import os

from composio.tools.base.local import LocalAction



class PDFEmbeddingRequest(BaseModel):
    directory: Optional[str] = Field(
        ...,
        description="Directory to scan pdf",
    )
    filepath: Optional[str] = Field(
        ...,
        description="Path of file to embed",
    )
    connection_string: str = Field(
        ...,
        description="Vector store connection string. Currently we are only supporting postgresql vector",
    )
    collection_name: str = Field(
        default='pdf_vector',
        description="Please provide a collection name if you have in mind. Default collection would be pdf_vector"
    )
    embedder: Literal["Ollama", "OpenAI", "Anthropic"]=Field(
        default="Ollama",
        description="Select a valid embedder default value will be ollama",
    )
    environment_key: str = Field(
        default="",
        description="If embeder is not ollama environment key required which stores api_key of models 'OPENAI_API_KEY'"
    )
    embedder_model: str = Field(
        ...,
        description="Model name of embedder"
    )
    model: str = Field(
        ...,
        description="Model name for query"
    )
    splitter_chunk_size: int = Field(
        default=500,
        description="Splitter chunk size default value will be 500. But you can pass based on your documents",
    )
    splitter_chunk_overlap: int = Field(
        default=50,
        description="Chunk overlap value for splitter default value will be 50. But you can change based on your requirements"
    )


    @model_validator(mode='after')
    def check_environment_key(self, model):
        if model.embedder != "Ollama" and not model.environment_key:
            raise ValueError(f"environment_key is required for embedders and query model")
        return model



class PDFEmbeddingResponse(BaseModel):
    retriever: VectorStoreRetriever


class PDFEmbedding(LocalAction[PDFEmbeddingRequest, PDFEmbeddingResponse]):
    """
    Embed pdf file or files nested in dir
    """

    def execute(self, request: PDFEmbeddingRequest, metadata: Dict) -> PDFEmbeddingResponse:
        file_path = request.filepath
        directory = request.directory
        splitter_chunk_size = request.splitter_chunk_size
        splitter_chunk_overlap = request.splitter_chunk_overlap

        if not dir and not file_path:
            raise ValueError(f"Error: you should provide a valid file path or a dir which contains the pdfs")
        
        pdf_files = []
        if file_path:
            pdf_files=[file_path]
        elif directory:
            pdf_files = self._extract_files_path(directory);
        
        if len(pdf_files)==0:
             raise ValueError(f"Error: Directory does not have any pdf files for embedding")
        
        loaded_docs  = []
        # load pdf files data
        for file in pdf_files:
            docs= self._text_loader(file_path=file);
            loaded_docs.append(docs)
        
        splitter = self._get_text_splitter(chunk_overlap=splitter_chunk_overlap, chunk_size=splitter_chunk_size)

        all_docs = []
        for docs in loaded_docs:
            splitted_docs = splitter.split_documents(docs)
            for doc in splitted_docs:
                doc.metadata['hash'] = hashlib.sha256(doc.page_content.encode()).hexdigest() # This is to check vectorstore does not store duplicate embeddings

            all_docs.extend(splitted_docs)

        embeddings = self._get_embedder(embedder_model=request.embedder_model)

        vector_store = self._get_vector_store(collection_name=request.collection_name, connection_string=request.connection_string, embedder=embeddings);

        new_docs = []

        for docs in all_docs:
            
            for doc in docs:
                #check for duplicate embeddings
                existing = vector_store.similarity_search("", k=1, filter=({"hash": doc.metadata["hash"]}))
                if not existing:
                    new_docs.append(doc)

            if len(new_docs)>0:
                vector_store.add_documents(new_docs)
             

        return PDFEmbeddingResponse(retriever=vector_store.as_retriever())
    
    def _extract_files_path(self,directory:str):
            # validate path
            if directory.endswith("*.*"):
                 raise ValueError(f"Error: directory path is not valid")

            files_path = []            
            for root, dir, files in os.walk(directory):
                 for file in files:
                      if file.endswith("*.pdf"):
                           valid_path = os.path.join(root, file)
                           files_path.append(valid_path)

            return files_path
    
    def _text_loader(self, file_path:str):
         
        try:
            from langchain_community.document_loaders.pdf import PyMuPDFLoader
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("The 'langchain_community' and 'pymupdf' is required for pdf load. Please install it using 'pip install langchain-community' and 'pip install pymupdf'") from e

        loader = PyMuPDFLoader(file_path=file_path, mode="page")
        docs = loader.load()
        return docs

    def _get_text_splitter(self, chunk_size:int, chunk_overlap:int):
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"The 'langchain' is required for text splitting. Please install it using 'pip install langchain'") from e
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        return splitter
    
    def _get_vector_store(self, connection_string: str, embedder: Any, collection_name:str):
        try:
            from langchain_postgres import PGVector
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"The 'langchain_postgres' package is required for vector store. Please install it using 'pip install -U langchain-postgres'") from e

        vector_store = PGVector(
                embeddings=embedder,
                collection_name=collection_name,
                connection=connection_string
            )
        
        return vector_store
    

    def _get_embedder(self, embedder_model: str):
        try:
            from langchain_ollama import OllamaEmbeddings
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"The 'langchain_ollama' package is required for embedding. Please install it using 'langchain-ollama'") from e
        
        if not embedder_model:
            raise ValueError(f"Embedder model name is required for ollama embeddings to use.")
        
        embeddings = OllamaEmbeddings(
            model=embedder_model
        )

        return embeddings