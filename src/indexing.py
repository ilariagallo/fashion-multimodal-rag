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

    def add_documents_from_file(
            self, file_name: str, page_content_fields: list[str], id_field: str, metadata_fields: list = None
    ):
        """
        Adds documents from a csv file to the vector store.

        :param file_name: csv file with documents to be added
        :param page_content_fields: list of fields to include in the text content of the document
        :param id_field: name of the field in the csv file that contains the document ID
        :param metadata_fields: list of metadata fields to be added to each document (optional)
        """
        items = pd.read_csv(file_name, dtype=str)

        # Add items to the vector store
        docs = [Document(page_content=self.format_document_content(item, page_content_fields),
                         metadata={m: item[m] for m in metadata_fields} if metadata_fields else {},
                         id=item[id_field]) for _, item in items.iterrows()]

        _ = self.vector_store.add_documents(documents=docs)

    @staticmethod
    def format_document_content(item: pd.Series, page_content_fields: list[str]):
        """
        Formats the content of the document used to generate embeddings.

        :param item: a single row from the dataframe
        :param page_content_fields: list of fields to include in the text content of the document
        :return: formatted string with the content of the document
        """
        return " ".join(f"<{field}>: {item[field]}" for field in page_content_fields)


if __name__ == "__main__":
    index = MultiModalIndex(embeddings=config.open_ai_embeddings)
    # Add documents from file
    index.add_documents_from_file(
        file_name=config.DATA_DIR + "articles_test.csv",
        page_content_fields=["prod_name", "product_group_name", "colour_group_name", "detail_desc"],
        id_field="article_id",
        metadata_fields=["prod_name", "product_type_name", "product_group_name", "colour_group_name"]
    )
