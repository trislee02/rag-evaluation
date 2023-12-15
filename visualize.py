import matplotlib.pyplot as plt
import pandas as pd
import os

EVALUATION_RESULT_FILE = "result/evalset40-history-ragas.csv"
OUTPUT_DIR = "figure/"

PAIRS = [["faithfulness", "context_recall"],
         ["faithfulness", "answer_similarity"],
         ["faithfulness", "request_recall"],
         ["context_recall", "request_recall"],
         ["answer_relevancy", "request_recall"],
        ]

df = pd.read_csv(EVALUATION_RESULT_FILE)

df['faithfulness'] = df['faithfulness'].astype(float)
df['context_recall'] = df['context_recall'].astype(float)
df['request_recall'] = df['request_recall'].astype(float)
df['answer_similarity'] = df['answer_similarity'].astype(float)
df['answer_relevancy'] = df['answer_relevancy'].astype(float)
for pair in PAIRS:   

    plt.xlabel(pair[0])
    plt.ylabel(pair[1])
    plt.plot(df[pair[0]], df[pair[1]], 'ro')

    output_filename = os.path.join(OUTPUT_DIR, f"FIG_{pair[0]}_{pair[1]}.png")

    plt.savefig(output_filename, bbox_inches='tight')
    plt.clf()
