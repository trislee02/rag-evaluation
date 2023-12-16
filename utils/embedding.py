import openai
import os
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

load_dotenv()

class Embedding:
    def __init__(self):
        openai.api_type = "azure"
        openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")
        openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.embedding_engine = os.getenv("AZURE_EMBEDDING_ENGINE")

    def get_embedding(self, text):
        return openai.Embedding.create(input=text, engine=self.embedding_engine)['data'][0]['embedding']

    def embedding_distance(self, embedding_1, embedding_2):
        return cosine(embedding_1, embedding_2) # cosine_distance
    
    def semantic_similarity(self, text_1, text_2):
        embed_1 = self.get_embedding(text_1)
        embed_2 = self.get_embedding(text_2)
        return 1 - self.embedding_distance(embed_1, embed_2)