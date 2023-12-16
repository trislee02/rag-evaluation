import os
import json
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from engine.question_generator import QuestionGenerator
from utils.embedding import Embedding
from dotenv import load_dotenv

import warnings 
warnings.filterwarnings('ignore') 

"""
This script is used to generate similar messages from given message. The input source is a directory containing .csv files with two column `message` and `response` as ground-truth for generation.

Usage:
    python generate_dataset.py -s sample_data/ -o synthesis_data.csv --generations 2

The `data` directory must contain .csv files:
    data
      | 01.csv
      | 02.csv
      | ...

The .csv contains pairs of message and response:
"message","response"
"how are you","I'm fine"
"hello","hi"
...

"""

question_gen = QuestionGenerator()
embed = Embedding()

COLUMN_NAMES = ["id", "message", "ground_message", "ground_answer", "message_similarity", "conversation_history"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True, help="Path to directory storing conversation .csv files")
    parser.add_argument("-o", "--output", required=True, help="Output file name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    parser.add_argument("--generations", default=3, type=int, help="Number of generations per sample")
    parser.add_argument("--maxdistance", default=1.0, type=float, help="A cosine distance threshold that triggers a new generation of message if the original message and the previous generated message are too different. Set it to 1.0 to skip regeneration.")
    parser.add_argument("--maxtries", default=3, type=int, help="A message can be regenerated up to maxtries times. If it goes beyond that, skip that message.")
    args = parser.parse_args()

    df_gen = pd.DataFrame([], columns=COLUMN_NAMES)

    try:
        file_no = 0
        for filepath in tqdm(glob.glob(os.path.join(args.source, "*.csv"))):
            file_no += 1

            df = pd.read_csv(filepath)
            df.fillna('', inplace=True)

            filename = os.path.basename(filepath)
            message_no = 0
            history = []
            for index, row in df.iterrows():
                message_no += 1
                message = row["message"]
                response = row["response"]
                history.append({"user": message, "bot": response})

                # Check whether both message and response are not empty
                if message != "" and response != "": 
                    sample_count = 0
                    sample_no = 0
                    while (sample_count < args.generations):
                        distance = 1.0
                        tries = 0
                        sample_count += 1
                        
                        generated_message = question_gen.generate(message)
                        while distance > args.maxdistance and tries <= args.maxtries:
                            if tries > 0:
                                if args.verbose: print(f"Too different from original request: Distance: {distance}")
                                generated_message = question_gen.regenerate(previous_generated_request=generated_message,
                                                                            sample_request=message)
                            distance = 1 - embed.semantic_similarity(generated_message, message)  
                            tries += 1               

                        # Check whether the generated message is valid
                        if tries <= args.maxtries:
                            sample_no += 1
                            message_id = f"{file_no:02d}.{message_no:02d}.{sample_no:02d}"
                            similarity = embed.semantic_similarity(generated_message, message) 
                            history_dumped = [json.dumps(conv) for conv in history[:-1]] # Exclude the current request
                            new_generation = {
                                "id": [message_id],
                                "message": [generated_message],
                                "ground_message": [message],
                                "ground_answer": [response],
                                "message_similarity": [similarity],
                                "conversation_history": [history_dumped]
                            }
                            df_new_row = pd.DataFrame(new_generation)
                            df_gen = pd.concat([df_gen, df_new_row], ignore_index=True, axis=0)
    except Exception as error:
        print(error)
    
    df_gen.to_csv(args.output, index=False)
    