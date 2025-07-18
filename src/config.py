import os

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

DATA_DIR = "../data/"
CHECKPOINTS_DIR = DATA_DIR + "checkpoints/checkpoints.sqlite"

load_dotenv()

open_ai_embeddings = AzureOpenAIEmbeddings(model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"])

chat = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


# Multi-modal open source embeddings. This requires the OpenCLIP model to be downloaded.
# This approach allows for image and text embeddings to be used together.
# However, it requires the OpenCLIP model to be downloaded and is not optimised for inference speed.

# clip_embeddings = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")
