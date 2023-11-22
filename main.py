import os
import csv
import glob
from engine.question_generator import QuestionGenerator
from utils.embedding import Embedding
from dotenv import load_dotenv
from engine.chatbot import Chatbot
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
    AnswerCorrectness
)
from ragas.metrics.critique import harmfulness

load_dotenv()

CHAT_ENDPOINT = os.environ.get("CHAT_ENDPOINT")

MAX_TRY = 3
DISTANCE_THRESHOLD = 0.05
NUMBER_QUESTIONS = 30

GEN_QUESTION_FILEPATH = 'synthetic_data/gen_data_10.csv'
EVALUATION_DATASET_FILEPATH = 'eval_dataset/eval_set_10.csv'
GROUND_TRUTH_DATA_PATH = './luxai-support-data-csv/*.csv'
EVALUATION_RESULT_FILEPATH = "result/evaluation_ragas.csv"

question_gen = QuestionGenerator()
embed = Embedding()
mybot = Chatbot(CHAT_ENDPOINT)

######################
#  GENERATE DATASET  #
######################

with open(GEN_QUESTION_FILEPATH, mode="w", newline='') as gen_dataset: 
    csv_writer = csv.writer(gen_dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    count = 0
    for filepath in glob.glob(GROUND_TRUTH_DATA_PATH):
        with open(filepath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            filename = os.path.basename(filepath)
            for row in csv_reader:
                if line_count > 0: # Omit the header 
                    count += 1
                    print(f"\nRequest #{count} from {filename}")
                    customer_request = row[0].strip()
                    luxai_support = row[1].strip()

                    if customer_request != "" and luxai_support != "": # Must be a pair of request and support
                        distance = 1
                        tries = 0
                        while distance > DISTANCE_THRESHOLD and tries < MAX_TRY:
                            if tries == 0:
                                generated_request = question_gen.generate(customer_request)
                            else:
                                generated_request = question_gen.regenerate(generated_request, customer_request)
                            distance = 1 - embed.semantic_similarity(generated_request, customer_request)  

                            print(f"\nTry #{tries} - Distance: {distance}")
                            if (distance > DISTANCE_THRESHOLD):
                                print("\nToo different from original request!")
                                print(generated_request)
                            tries += 1               
                        if tries < MAX_TRY:
                            csv_writer.writerow([generated_request, customer_request, luxai_support])
                    else:
                        print("Pair missing!")
                line_count += 1

######################
#   GENERATE ANSWER  #
######################

with open(OUTPUT_FILE, mode="w", newline='') as gen_dataset: 
    csv_writer = csv.writer(gen_dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    columns = ['question','answer','ground_truths','contexts']
    csv_writer.writerow(columns)
    count = 0
    with open(DATASET_FILE) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            count += 1
            print(f"Row #{count}")
            if len(row) <= 0 or row[0].strip() == "":
                continue
            question = row[0]
            ground_request = row[1]
            ground_response = row[2]
            
            response = mybot.run(question)
            
            answer = response["answer"]
            contexts = ['']
            if "contexts" in response:
                contexts = response["contexts"]

            ground_truths = [ground_response]

            csv_writer.writerow([question, answer, ground_truths, contexts])

############################
#   Evaluate using RAGAS   #
############################
metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    harmfulness,
]

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
os.environ["OPENAI_API_BASE"] = "https://tri-testing-3.openai.azure.com/"
# os.environ["OPENAI_API_KEY"] = "" # Set the OPENAI_API_KEY in the .env file

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
DATASET_FILE = EVALUATION_DATASET_FILEPATH
df = pd.read_csv(DATASET_FILE)
df['contexts'] = df.contexts.apply(lambda x: ast.literal_eval(x))
df['ground_truths'] = df.ground_truths.apply(lambda x: ast.literal_eval(x))

dataset = Dataset.from_pandas(df)
print(dataset)

results = evaluate(dataset, metrics=metrics)
print(results)

df = results.to_pandas()
df.to_csv(EVALUATION_RESULT_FILEPATH)
print(df.head())