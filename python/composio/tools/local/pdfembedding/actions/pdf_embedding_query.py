from pydantic import BaseModel, Field
from langchain_core.vectorstores import VectorStoreRetriever



class PDFEmbeddingQueryRequest(BaseModel):
    retriever: VectorStoreRetriever = Field(
        ...,
        description="retriever is required for query"
    )
    query: str = Field(
        ...,
        description="Embedding query"
    )

class PDFEmbeddingQueryResponse(BaseModel):
    context: str = Field(
        ...,
        description="Retrieved documents from vector store"
    )

class PDFEmbeddingQuery(LocalAction[PDFEmbeddingQueryRequest, PDFEmbeddingQueryResponse]):
    """
    To Retrieved context of query for agent or llm
    """


    def execute(request: EmebddingQueryRequest, metadat: dict={}):
        retriever = request["retriever"]
        query = request["query"]

        if not retriever:
            raise ValueError(f"Retriever is required to fetch relevant documents.")
        
        if not query:
            raise ValueError(f"Please provide a valid query.")
        
        # Retrieved relevant document for context

        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        return PDFEmbeddingQueryResponse(context=context)

