import os
import csv
import glob
import json
import ast
import pandas as pd
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from engine.chatbot import Chatbot

"""
This script is used to send POST request to chatbot server to get answer. The input source is the output of running `generate_dataset.py`

Usage:
    python generate_answer.py -s synthesis_data.csv -o answers.csv     
"""

load_dotenv()
CHAT_ENDPOINT = os.environ.get("CHAT_ENDPOINT")
mybot = Chatbot(CHAT_ENDPOINT)

COLUMN_NAMES = ['question','answer','ground_truths','contexts','request','supporting_content','tokens_usage','conversation_history']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True, help="Path to message file")
    parser.add_argument('-o', '--output', required=True, help="Output answer file")
    args = parser.parse_args()

    df_out = pd.DataFrame([], columns=COLUMN_NAMES)
    df_in = pd.read_csv(args.source)
    df_in.fillna('', inplace=True)

    try:
        for index, row in tqdm(df_in.iterrows(), total=df_in.shape[0]):
            message = row["message"]
            ground_response = row["ground_answer"]
            history = row["conversation_history"]
            
            parsed_history = ast.literal_eval(history)
            parsed_history = [json.loads(element) for element in parsed_history]
            response = mybot.run(message, history=parsed_history)
            
            answer = response["answer"]
            contexts = []
            if "contexts" in response:
                contexts = response["contexts"]
            supporting_contents = response["supporting_contents"]
            token_usage = response["token_usage"]
            extracted_requests = response["extracted_requests"]
            ground_truths = [ground_response]

            extracted_requests = "\n".join(extracted_requests)        
            contexts.append(supporting_contents)
        
            new_answer = {
                "question": [message],
                "answer": [answer],
                "ground_truths": [ground_truths],
                "contexts": [contexts],
                "request": [extracted_requests],
                "supporting_content": [supporting_contents],
                "tokens_usage": [token_usage],
                "conversation_history": [history]
            }
            df_out = pd.concat([df_out, pd.DataFrame(new_answer)], ignore_index=True)
    except Exception as error:
        print(error)

    df_out.to_csv(args.output, index=False)    