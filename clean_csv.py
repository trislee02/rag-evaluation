import json
import ast
import pandas as pd

def clean_cell(x):
    if type(x) is str:
        x = x.replace("\\n", "\n").replace("\\t", "\t").replace("\\r","").replace("\n\n", "\n")
    return x

INPUT_FILE = "result/evalset100-summary-vectors-ragas-2.csv"
OUTPUT_FILE = "result/evalset100-summary-vectors-ragas-2-cleaned.csv"
df = pd.read_csv(INPUT_FILE)

df = df.map(clean_cell)

df.to_csv(OUTPUT_FILE, index=False)