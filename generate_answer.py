import os
import csv
import glob
import json
import ast

from dotenv import load_dotenv
from engine.chatbot import Chatbot

load_dotenv()

CHAT_ENDPOINT = os.environ.get("CHAT_ENDPOINT")

mybot = Chatbot(CHAT_ENDPOINT)

OUTPUT_FILE = 'eval_dataset/answer_humangen_dataset_reranker.csv'
# DATASET_FILE = 'synthetic_data/data30-history.csv'
DATASET_FILE = 'synthetic_data/humangen_dataset.csv'

with open(OUTPUT_FILE, mode="w", newline='', encoding="utf-8") as gen_dataset: 
    csv_writer = csv.writer(gen_dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    columns = ['question','answer','ground_truths','contexts','request','supporting_content','tokens_usage','conversation_history']
    csv_writer.writerow(columns)
    count = -1
    with open(DATASET_FILE) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        line_count = 0
        for row in csv_reader:
            count += 1
            if count < 1:
                continue
            print(f"Row #{count}")
            if len(row) <= 0 or row[0].strip() == "":
                continue
            question_id = row[0]
            question = row[1]
            ground_request = row[2]
            ground_response = row[3]
            history = row[5]

            # history = history.replace("\\r\\n", "\n").replace("\\n", '\n').replace("\\'", '\\"')
            parsed_history = ast.literal_eval(history)
            parsed_history = [json.loads(element) for element in parsed_history]
            # print(history)
            # input()
            response = mybot.run(question, history=parsed_history)
            
            answer = response["answer"]
            contexts = []
            if "contexts" in response:
                contexts = response["contexts"]
            supporting_contents = response["supporting_contents"]
            token_usage = response["token_usage"]
            extracted_requests = response["extracted_requests"]


            ground_truths = [ground_response]

            # supporting_contents = supporting_contents.replace("\\r\\n", "\n").replace("\\r", '')
            extracted_requests = "\n".join(extracted_requests)
            
            contexts.append(supporting_contents)

            # Only record answers generated from specific contexts, i.e. the LLM knows the answer based on given context
            if len(contexts) > 0:
                csv_writer.writerow([question, answer, ground_truths, contexts, extracted_requests, supporting_contents, token_usage, history])
