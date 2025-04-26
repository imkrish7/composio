import os
import hashlib
from typing import Any, Dict, Literal, Optional
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import BaseModel, Field, model_validator


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
        description="Connection string for the vector store: We currently only support PostgreSQL for vector storage.",
    )
    collection_name: str = Field(
        default='pdf_vector',
        description="Please provide a collection name if you have one in mind. The default collection name is pdf_vector."
    )
    embedder: Literal["Ollama", "OpenAI"]=Field(
        default="Ollama",
        description="Please provide a valid embedder. The default value is 'ollama'.",
    )
    embedding_model_environment_key: str = Field(
        default=None,
        description="Please provide your model API key if you are not using 'ollama'."
    )
    model: str = Field(
        ...,
        description="Embedding model name"
    )
    splitter_chunk_size: int = Field(
        default=500,
        description="Text splitter chunk size default value will be 500. But you can pass based on your use case.",
    )
    splitter_chunk_overlap: int = Field(
        default=50,
        description="By default, the chunk overlap value for the splitter is set to 50, but you can adjust it based on your specific use case."
    )


    @model_validator(mode='after')
    def check_environment_key(self, model):
        if model.embedder != "Ollama" and not model.embedding_model_environment_key:
            raise ValueError(f"embedding_model_environment_key is required for embedder model.")
        return model



class PDFEmbeddingResponse(BaseModel):
    retriever: VectorStoreRetriever


class PDFEmbedding(LocalAction[PDFEmbeddingRequest, PDFEmbeddingResponse]):
    """
    Embed pdf file or files nested in dir
    """

    def execute(self, request: PDFEmbeddingRequest, metadata: Dict) -> PDFEmbeddingResponse:
        file_path = request.filepath or metadata["filepath"]
        directory = request.directory or metadata["directory"]
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
             raise ValueError(f"Error: Directory does not have any pdf file to embed")
        
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

        embeddings = self._get_embedder(embedder_model=request.model, embedder=request.embedder, environment_key=request.embedding_model_environment_key)

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
    

    def _get_embedder(self, embedder_model: str, embedder: str, environment_key: str):

        embeddings = None

        if embedder == 'OpenAI':
            try:
                from langchain_openai import OpenAIEmbeddings
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(f"The 'langchain-openai' package is required for embedding. Please install it using 'pip install -qU langchain-openai'")
            
            api_key = os.environ.get(environment_key)

            if not embedder_model:
                raise ValueError(f"Embedder model name is required for openai embeddings to use.")

            embeddings = OpenAIEmbeddings(
                model=embedder_model,
                api_key=api_key
            )
        else:
            try:
                from langchain_ollama import OllamaEmbeddings
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(f"The 'langchain-ollama' package is required for embedding. Please install it using 'langchain-ollama'") from e

            if not embedder_model:
                raise ValueError(f"Embedder model name is required for ollama embeddings to use.")

            embeddings = OllamaEmbeddings(
                model=embedder_model
            )
        

        return embeddings