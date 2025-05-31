import pandas as pd
import config

from langchain_core.documents import Document
from langchain_chroma import Chroma

class MultiModalIndex:
    """Implements a multimodal index to store and retrieve document and image embeddings efficiently"""

    COLLECTION_NAME = "fashion_products"
    DIRECTORY_NAME = config.DATA_DIR + "chroma_langchain_db"

    def __init__(self, embeddings=None, vector_store=None):
        """
        Constructor for the in-memory index.

        :param embeddings: embedding function, defaults to Azure OpenAI embeddings
        :param vector_store: vector store implementation, defaults to ChromaDB
        """
        self.embeddings = embeddings or config.open_ai_embeddings
        self.vector_store = vector_store or Chroma(collection_name=self.COLLECTION_NAME,
                                                   embedding_function=self.embeddings,
                                                   persist_directory=self.DIRECTORY_NAME)

    def add_documents_from_file(self, file_name: str, page_content_field: str, id_field: str, metadata_fields: list = None):
        """
        Adds documents from a csv file to the vector store.

        :param file_name: csv file with documents to be added
        :param page_content_field: name of the field in the csv file that contains the text content of the document
        :param id_field: name of the field in the csv file that contains the document ID
        :param metadata_fields: list of metadata fields to be added to each document (optional)
        """
        items = pd.read_csv(file_name, dtype=str)

        # Add items to the vector store
        docs = [Document(page_content=item[page_content_field],
                         metadata={m: item[m] for m in metadata_fields} if metadata_fields else {},
                         id=item[id_field]) for _, item in items.iterrows()]

        _ = self.vector_store.add_documents(documents=docs)


if __name__ == "__main__":
    index = MultiModalIndex(embeddings=config.open_ai_embeddings)
    # Add documents from file
    index.add_documents_from_file(
        file_name=config.DATA_DIR + "articles_test.csv",
        page_content_field="detail_desc",
        id_field="article_id",
        metadata_fields=["prod_name", "product_type_name", "product_group_name"]
    )

    # print(index.vector_store.similarity_search('t-shirt top with short sleeves', k=1))
    # print(index.vector_store.similarity_search('long sleeve sweater', k=1))

    # image_path = '/Users/u022ixg/PycharmProjects/fashion-multimodal-rag/data/images/0237347017.jpg'
    # print(index.vector_store.similarity_search_by_image(image_path, k=2))
