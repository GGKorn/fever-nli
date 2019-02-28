# ANLP Final Project: Claim Verification

Uni Potsdam WiSe 2018/19
Dr. Tatjana Scheffler

Group Members:
Gunnar Gerstenkorn
Benjamin Henne
Alexandra Horsley

## Project Summary
This project is largely based on the Fact Extraction and Verification (FEVER) challenge, the first iteration of which occurred in late 2018. This implementation is a decomposable attention neural network (a variant of Bahdanau et al.'s Neural Machine Translation model) that predicts textual entailment resulting from multiple pieces of evidence. Given a claim and a number of sentences or other pieces of information from Wikipedia articles (hyperlinks, the title of the source article), this neural network infers whether the given piece(s) of evidence support(s) or refute(s) the given claim, or if there is not enough information given.

We used word embeddings, sentence alignment, and other linguistic features to perform this natural language inference task, and this particular implementation has a feed-forward structure.

This model also includes the FEVER baseline model as a means to provide context/a meaningful comparison between the results.

Further information can be found in the forthcoming project reports.

## Results
This model was tested against two types of training data: the vanilla variation only contains the evidence sentences given, and the extended variation includes the article title, tags appended to the end of each sentence, and additional context sentences (those surrounding the given piece of evidence, as well as the first sentence of the given text, if these are not already included in evidence). These achieved the following accuracies:
### FEVER Baseline Model:
- Vanilla:
- Extended:
### DA Model:
- Vanilla:
- Extended:


## Implementing the Model

### Step 1: Download and install necessary items
1. Run `git clone https://gitup.uni-potsdam.de/ANLP_Claim_Verification/anlp_final_project.git`
2. `cd anlp_final_project`
3. Download all packages needed if not currently present on your machine: `pip install -r required_packages.txt`

### Step 2 (Optional): Compose the datasets
Alternatively, one could use the pre-cleaned data in the existing `extended_wiki_data` and `vanilla_wiki_data` folders.
1. `cd data`
2. Download the [fever.ai dataset](https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl) and [June 2017 Wikipedia dump](https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip) to the `data` directory
3. `python compose_dataset_vanilla.py`
4. `python compose_dataset_extended.py`

*This will yield a total of six files. If any of them are too large for any reason, one could use the split_data.py protocol*

### Step 3: Parse the data and implement the baseline model
From the main anlp_final_project/src directory
```
python input_fnc.py
python model_fnc.py
python main.py
```

### Step 4: Parse the data and implement the DA model
From the anlp_final_project/src directory
```
python input_da.py
python model_da.py
python main.py
```
