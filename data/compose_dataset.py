import json
import os
import csv
from collections import defaultdict

# path to wiki-dump files
path_to_data = r"E:/Data/wikipedia_dump_2017/wiki-pages/"
# collect the names of all wiki files
json_files = [fname for fname in os.listdir(path_to_data) if fname.startswith('wiki-') and fname.endswith('.jsonl')]

# helper functions to process text
def read_lines(article):
    return [line for line in article.split("\n")]

def normalize(text):
    """Resolve different types of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def read_text(article):
    return [normalize(line.split('\t')[1]) if len(line.split('\t'))>1 else "" for line in read_lines(article)]



# compile wiki files into one massive dictionary
wiki_data = defaultdict(list)
for i,f in enumerate(json_files):
    with json.open(os.path.join(path_to_data, f)) as jreader:
        for itm in jreader:
            for article in itm:
                wiki_data[article['id']] = read_text(article['lines'])
            print("Parsing {} of {} data chunks. Total entries: {}".format(i+1, len(json_files), len(wiki_data)))

# create csv to write to
with open('cleaned_wiki_data.csv', 'wb+') as data:
    csv_data = csv.writer(data)
    csv_data.writerow(['id', 'verifiable', 'label', 'claim', 'evidence'])

    # collect evidence ids and pair to wiki_data
    evidence_dict = itertools(list)
    with json.open('./data/train.jsonl') as jreader:
        for itm in jreader:
            print("itm: ", itm)
            evidence_dict[itm['id']] = []
            for evidence in itm['evidence']:
                for anno_id, evidence_id, article_name, sentence_id in evidence:
                    evidence_dict[itm['id']].append(wiki_data[article_name][sentence_id])
                    print("evidence_id:(", evidence_id, ")\narticle_name:(", article_name, ")\nsentence_id:(", sentence_id, ")")
            csv_data.writerow([itm['id'], itm['verifiable'], itm['label'], itm['claim'], evidence_dict[itm['id']]])
