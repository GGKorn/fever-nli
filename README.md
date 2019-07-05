# ANLP Final Project: Claim Verification

Uni Potsdam WiSe 2018/19

Dr. Tatjana Scheffler

Group Members:
Gunnar Gerstenkorn, Benjamin Henne, Alexandra Horsley

## Project Summary
This project is largely based on the Fact Extraction and Verification (FEVER) challenge, the first iteration of which occurred in late 2018. This implementation is a decomposable attention neural network (proposed by [Parikh et al. in 2016](https://arxiv.org/abs/1606.01933), utilising [Luong Attention (2015)](https://arxiv.org/abs/1508.04025) to perform neural machine translation) that predicts textual entailment resulting from multiple pieces of evidence. Given a claim and a number of sentences or other pieces of information from Wikipedia articles (hyperlinks, the title of the source article), this neural network infers whether the given piece(s) of evidence support(s) or refute(s) the given claim, or if there is not enough information given.

We used word embeddings, sentence alignment, and other linguistic features to perform this natural language inference task, and this particular implementation has a feed-forward structure.

This model also includes the FEVER baseline model as a means to provide context/a meaningful comparison between the results.

Further information can be found in the forthcoming project reports.

## Results
This model was tested against two types of training data: the vanilla variation only contains the evidence sentences given, and the extended variation includes the article title, tags appended to the end of each sentence, and additional context sentences (those surrounding the given piece of evidence, as well as the first sentence of the given text, if these are not already included in evidence). These achieved the following accuracies:

| FEVER Baseline Model &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;| Accuracy |
|----------------------|----------|
| Vanilla              | 0.6681   |
| Extended             | 0.6646   |

| Decomposable Attention Model | Accuracy |
|------------------------------|----------|
| Vanilla                      | 0.7948   |
| Extended                     | 0.7943   |


## Setting up the Model
Please note that this process can be quite lengthy. A 64-bit version of Python 3.5.4 is required.

### Step 1: Download and install necessary items
1. Clone the repository: `https://github.com/benjaminhenne/fever-nli.git`
2. `cd fever-nli`
3. Supply your Python environment with all packages needed that are not currently present on your machine: `pip install -r required_packages.txt`

### Step 2 (Optional): Compose the datasets
Alternatively, one could use the pre-cleaned data in the existing `extended_wiki_data` and `vanilla_wiki_data` folders. These files have already undergone the instructions of Step 2.
1. `cd data`
2. Download the [fever.ai dataset](https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl): `wget https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl`
3. Download and unzip the [June 2017 Wikipedia dump](https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip): `wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip`
3. `python compose_dataset_vanilla.py`
4. `python compose_dataset_extended.py`

*This will yield a total of six files. If any of them are too large for any reason, one could use the split_data.py protocol (this will prompt you for the file you wish to split, the folder these split files will be directed to, the prefix you want on each of these new files, and the number of lines you wish to have in each split of the file)*

### Step 3: Download and preprocess the word embeddings
1. `mkdir embeddings && cd embeddings`
2. Download and unzip the [GloVe (6B tokens, 400k vocab, 50d-300d) word embeddings](http://nlp.stanford.edu/data/glove.6B.zip): `wget http://nlp.stanford.edu/data/glove.6B.zip`
3. Convert the GloVe embedding to gensim-compatible KeyedVectors format: `python -m gensim.scripts.glove2word2vec -i glove.6B.200d.txt -o gensim_glove.6B.200d.txt`
4. Repeat this process for other dimensions of the GloVe embeddings you plan to use.
5. Open config file `anlp_claim_verification/config/config.yaml` and enter the absolute file path of the embedding and its dimension `(50, 100, 200, 300)`. This specifies which embedding will be used by the model during execution.

### Step 4: Train the baseline mode and evaluate its performance
From the root directory `fever-nli`:
```
python src/main.py -m 1 -i data/vanilla_wiki_data -o results/ -b 500 -e 500 -l 0.01 -s 5000 -j <..> -a <..>
python src/main.py -m 1 -i data/extended_wiki_data -o results/ -b 500 -e 500 -l 0.01 -s 5000 -j <..> -a <..>
```
Each of these commands will start the training of the baseline (`-m 1`) model, using either the vanilla dataset (`-i data/vanilla_wiki_data`) or the extended dataset (`-i data/extended_wiki_data`), depositing results, checkpoints, and graph summaries into a corresponding folder in results (`-o results/`). 

Specific hyperparameters (our configuration) for the run will include a batch size of 500 (`-b 500`), an evaluation batch size of 500 (`-e 500`), and a learning rate of 0.01 (`-l 0.01`). Furthermore, a user-selected job-id needs to be provided (`-j <..>`), which will be used to identify the run in the directory structure of the results. Repetitions (array jobs) of the same job can be requested by supplying `-a` followed by any value larger than 1.

### Step 5: Train the Decomposable Attention mode and evaluate its performance
From the root directory `fever-nli`:
```
python src/main.py -m 2 -i data/vanilla_wiki_data -o results/ -b 32 -e 2000 -l 0.05 -s 5000 -j <..> -a <..>
python src/main.py -m 2 -i data/extended_wiki_data -o results/ -b 32 -e 2000 -l 0.05 -s 5000 -j <..> -a <..>
```
This will begin the training of the decomposable attention model. The batch size has been increased from 4 as proposed in the paper to 32 to provide a more accurate representation of the dataset as a whole. The overall process of manufacturing batches is less costly for this model, so the evaluation batch size could be increased substantially as well to provide more accurate evaluation metrics. Other hyperparameters remain unchanged from Step 4.
