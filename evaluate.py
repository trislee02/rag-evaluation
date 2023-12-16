import os
import pandas as pd
import json
import ast
import argparse
import openai
from ragas import evaluate
from datasets import Dataset
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from ragas.llms import LangchainLLM

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    request_recall,
)
from ragas.metrics import AnswerSimilarity
from dotenv import load_dotenv

"""
This script is used to evaluate the answer output after runing `generate_answer.py` using RAGAS framework.

Usage:
    python evaluate.py -s answers.csv -o evaluation_result.csv     
"""

def parse_history(history: str) -> list[dict]:
    parsed_history = ""
    for history_item in history:
        conv = json.loads(history_item)
        parsed_history += "user:" + conv.get("user") + "\n"
        if "bot" in conv:
            parsed_history += "assistant:" + conv.get("bot") + "\n"
    return parsed_history

def clean_cell(x):
    if type(x) is str:
        x = x.replace("\\n", "\n").replace("\\t", "\t").replace("\\r","").replace("\n\n", "\n")
    return x

def clean_contexts(x):
    # for i in range(len(x)):
    #     if len(x[i]) > 1000:
    #         x[i] = x[i][:1000]
    #         print(len(x[i]))
    return x

load_dotenv()
openai.api_type = "azure"
openai.api_base = os.environ.get("AZURE_OPENAI_GPT4_API_BASE")
openai.api_version = os.environ.get("AZURE_OPENAI_GPT4_API_VERSION")
openai.api_key = os.environ.get("AZURE_OPENAI_GPT4_API_KEY")

my_answer_similarity = AnswerSimilarity(threshold=None)

metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    request_recall,
    my_answer_similarity,
]

azure_model = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_GPT4_CHAT_DEPLOYMENT"),
    model="gpt-4",
    openai_api_base=os.getenv("AZURE_OPENAI_GPT4_API_BASE"),
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
    deployment=os.getenv("AZURE_GPT4_EMBEDDING_ENGINE"),
    model="text-embedding-ada-002",
    openai_api_base=os.getenv("AZURE_OPENAI_GPT4_API_BASE"),
    openai_api_type="azure",
)
# embeddings can be used as it is
answer_relevancy.embeddings = azure_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--source", required=True, help="Path to answer file")
    parser.add_argument('-o', "--output", required=True, help="Output file name")
    parser.add_argument('-d', "--debug", action="store_true", help="Run evaluation with a few answers for debugging purpose")
    args = parser.parse_args()

    df = pd.read_csv(args.source)

    df['contexts'] = df.contexts.apply(lambda x: clean_contexts(ast.literal_eval(x)))
    df['ground_truths'] = df.ground_truths.apply(lambda x: ast.literal_eval(x))
    df['conversation_history'] = df.conversation_history.apply(lambda x: parse_history(ast.literal_eval(x)))

    if args.debug:
        print("Debug mode")
        df = df.head(2)
        
    dataset = Dataset.from_pandas(df)
    print(dataset)

    results = evaluate(dataset, metrics=metrics, verbose=True)
    print(results)

    df = results.to_pandas()

    df = df.map(clean_cell)

    df.to_csv(args.output, index=False)
    print(df.head())