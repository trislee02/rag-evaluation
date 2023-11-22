import os
import pandas as pd
import ast
from ragas import evaluate
from datasets import Dataset
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLM

from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    context_recall,
    answer_similarity,
    answer_correctness,
    request_recall,
)

from ragas.metrics import AnswerSimilarity

from ragas.metrics.critique import harmfulness

my_answer_similarity = AnswerSimilarity(threshold=None)

metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    request_recall,
    my_answer_similarity,
]

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ["OPENAI_API_BASE"] = "https://tri-testing-3.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "b805bc4be3834e7abaab768405ffde29"

azure_model = AzureChatOpenAI(
    deployment_name="tri-gpt-4",
    model="gpt-4",
    openai_api_base="https://tri-testing-3.openai.azure.com/",
    openai_api_type="azure",
)
# wrapper around azure_model 
ragas_azure_model = LangchainLLM(azure_model)
# patch the new RagasLLM instance
answer_relevancy.llm = ragas_azure_model

# init and change the embeddings
# only for answer_relevancy
azure_embeddings = OpenAIEmbeddings(
    deployment="tri-ada-canada",
    model="text-embedding-ada-002",
    openai_api_base="https://tri-testing-3.openai.azure.com/",
    openai_api_type="azure",
)
# embeddings can be used as it is
answer_relevancy.embeddings = azure_embeddings

for m in metrics:
    m.__setattr__("llm", ragas_azure_model)

# prepare your huggingface dataset in the format
DATASET_FILE = "eval_dataset/evalset30-history.csv"
OUTPUT_EVALUATION_FILE = "result/evalset30-history-ragas.csv"
df = pd.read_csv(DATASET_FILE)

df['contexts'] = df.contexts.apply(lambda x: ast.literal_eval(x))
df['ground_truths'] = df.ground_truths.apply(lambda x: ast.literal_eval(x))


dataset = Dataset.from_pandas(df)
print(dataset)

results = evaluate(dataset, metrics=metrics, verbose=True)
print(results)

df = results.to_pandas()
df.to_csv(OUTPUT_EVALUATION_FILE)
print(df.head())