import os
import pandas as pd
import json
import ast
import logging
from ragas import evaluate
from datasets import Dataset
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
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

# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)

def parse_history(history: str) -> list[dict]:
    parsed_history = ""
    for history_item in history:
        conv = json.loads(history_item)

        parsed_history += "user:" + conv.get("user") + "\n"
        if "bot" in conv:
            parsed_history += "assistant:" + conv.get("bot") + "\n"

    # print(parsed_history)
    return parsed_history

def clean_cell(x):
    if type(x) is str:
        x = x.replace("\\n", "\n").replace("\\t", "\t").replace("\\r","").replace("\n\n", "\n")
    return x

def clean_contexts(x):
    # for i in range(len(x)):
    #     if len(x[i]) > 10000:
    #         x[i] = x[i][:10000]
    #         print(len(x[i]))
    return x

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
for m in metrics:
    m.__setattr__("llm", ragas_azure_model)

# init and change the embeddings
# only for answer_relevancy
azure_embeddings = AzureOpenAIEmbeddings(
    deployment="tri-ada-canada",
    model="text-embedding-ada-002",
    openai_api_base="https://tri-testing-3.openai.azure.com/",
    openai_api_type="azure",
)
# embeddings can be used as it is
answer_relevancy.embeddings = azure_embeddings

# prepare dataset
DATASET_FILE = "eval_dataset/evalset100-summary-vectors.csv"
OUTPUT_EVALUATION_FILE = "result/evalset100-summary-vectors-ragas-2.csv"
DEBUGGING = False

df = pd.read_csv(DATASET_FILE)

df['contexts'] = df.contexts.apply(lambda x: clean_contexts(ast.literal_eval(x)))
df['ground_truths'] = df.ground_truths.apply(lambda x: ast.literal_eval(x))
df['conversation_history'] = df.conversation_history.apply(lambda x: parse_history(ast.literal_eval(x)))

if DEBUGGING:
    df = df.head(2)
    print("*************************")
    print("*       DEBUGGING       *")
    print("*************************")

dataset = Dataset.from_pandas(df)
print(dataset)

results = evaluate(dataset, metrics=metrics, verbose=True)
print(results)

df = results.to_pandas()

df = df.map(clean_cell)

df.to_csv(OUTPUT_EVALUATION_FILE, index=False)
print(df.head())