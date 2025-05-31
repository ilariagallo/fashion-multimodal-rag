import os

from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import vertexai

# PROJECT_ID = 'fashion-multimodal-rag'
# LOCATION= 'us-central1'
# vertexai.init(project=PROJECT_ID, location=LOCATION)
# vertex_embeddings = VertexAIEmbeddings(model_name="multimodalembedding", project='fashion-multimodal-rag')

DATA_DIR = "../data/"
CHECKPOINTS_DIR = DATA_DIR + "checkpoints/checkpoints.sqlite"

load_dotenv()

open_ai_embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
# Multi-modal open source embeddings. This requires the OpenCLIP model to be downloaded.
# This approach allows for image and text embeddings to be used together.
# However, it requires the OpenCLIP model to be downloaded and is not optimised for inference speed.

# clip_embeddings = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")
