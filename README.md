# RAG Evaluation

This repo contains scripts to evaluate the whole RAG pipeline using my customized [RAGAS framework](https://github.com/trislee02/ragas). 

# Usage

* Prepare a directory storing conversations. Each conversation is stored in a .csv file with ground-truth message-response pairs. Look at `sample_data` folder for reference.
* Run `generate_dataset.py` to generate more data for evaluation.
```
python generate_dataset.py -s sample_data/ -o synthesis_data.csv --generations 2
```
* Run `generate_answer.py` to get answers from Chatbot.
```
python generate_answer.py -s synthesis_data.csv -o answers.csv   
```
* Run `evaluate.py` to evaluate the Chatbot with RAGAS metrics.
```
python evaluate.py -s answers.csv -o evaluation_result.csv  
```
