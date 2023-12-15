import os
import csv
import json
import glob
from engine.question_generator import QuestionGenerator
from utils.embedding import Embedding

MAX_TRY = 3
DISTANCE_THRESHOLD = None 
NUMBER_GENERATIONS = 1

question_gen = QuestionGenerator()
embed = Embedding()

OUTPUT_SYNTHETIC_DATA = 'synthetic_data/data40-unseen.csv'
SOURCE_DIR = "./luxai-support-data-csv/*.csv"

with open(OUTPUT_SYNTHETIC_DATA, mode="w", newline='') as gen_dataset: 
    csv_writer = csv.writer(gen_dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["id", "request", "ground_request", "ground_answer", "request_similarity", "conversation_history"]) # Headers
    file_no = 0
    total_generated_request = 0
    for filepath in glob.glob(SOURCE_DIR):
        file_no += 1
        with open(filepath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            filename = os.path.basename(filepath)
            request_no = 0
            history = []
            for row in csv_reader:
                if line_count > 0: # Omit the header 
                    request_no += 1
                    customer_request = row[0].strip()
                    luxai_support = row[1].strip()
                    history.append({"user": customer_request, "bot": luxai_support})

                    if customer_request != "" and luxai_support != "": # Must be a pair of request and support
                        sample_count = 0
                        sample_no = 0
                        while (sample_count < NUMBER_GENERATIONS):
                            distance = 1
                            tries = 0
                            sample_count += 1
                            
                            print(f"\nRequest #{request_no} from {file_no} ({filename}), sample #{sample_count}")

                            generated_request = question_gen.generate(customer_request)
                            while DISTANCE_THRESHOLD and distance > DISTANCE_THRESHOLD and tries < MAX_TRY:
                                distance = 1 - embed.semantic_similarity(generated_request, customer_request)  
                                print(f"\nTry #{tries} - Distance: {distance}")

                                if (distance > DISTANCE_THRESHOLD):
                                    print("\nToo different from original request!")
                                    print(generated_request)
                                    generated_request = question_gen.regenerate(generated_request, customer_request)
                                    tries += 1               

                            if tries < MAX_TRY:
                                sample_no += 1
                                request_id = f"{file_no:02d}.{request_no:02d}.{sample_no:02d}"
                                similarity = embed.semantic_similarity(generated_request, customer_request) 
                                total_generated_request += 1 
                                print(f"Total generated requests = {total_generated_request}\n")
                                history_dumped = [json.dumps(conv) for conv in history[:-1]] # Exclude the current request
                                print(history_dumped)
                                csv_writer.writerow([request_id, generated_request, customer_request, luxai_support, similarity, history_dumped])
                    
                    else:
                        print("Pair missing!")
                line_count += 1
                