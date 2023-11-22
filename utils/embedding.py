import openai
from scipy.spatial.distance import cosine

class Embedding:
    def __init__(self):
        openai.api_type = "azure"
        openai.api_base = "https://tri-testing.openai.azure.com/"
        openai.api_version = "2023-05-15"
        openai.api_key = "302e2d25248b442faa27c9b213f3fb2d"

    def get_embedding(self, text, model="text-embedding-ada-002"):
        return openai.Embedding.create(input=text, engine="tri-ada")['data'][0]['embedding']

    def embedding_distance(self, embedding_1, embedding_2):
        return cosine(embedding_1, embedding_2) # cosine_distance
    
    def semantic_similarity(self, text_1, text_2):
        embed_1 = self.get_embedding(text_1)
        embed_2 = self.get_embedding(text_2)
        return 1 - self.embedding_distance(embed_1, embed_2)