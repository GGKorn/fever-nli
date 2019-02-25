import json
import os
import csv
#comment back in if uing the normalize function
#import unicodedata
from collections import defaultdict

def main():
    # path to wiki-dump files
    path_to_data = r"wiki-pages/"
    # collect the names of all wiki files
    json_files = [fname for fname in os.listdir(path_to_data) if fname.startswith('wiki-') and fname.endswith('.jsonl')]

    ### helper functions to process text -- not used currently, will probably use package to normalize/tokenize text
    # def read_lines(article):
    #     return [line for line in article.split("\n")]
    #
    # def normalize(text):
    #     """Resolve different types of unicode encodings."""
    #     return unicodedata.normalize('NFD', text)
    #
    # def read_text(article):
    #     return [normalize(line.split('\t')[1]) if len(line.split('\t'))>1 else "" for line in article.split('\n')]
    #


    # compile wiki files into one massive dictionary
    wiki_data = {}
    for i,f in enumerate(json_files):
        with open(os.path.join(path_to_data, f)) as jreader:
            for itm in jreader:
                j = json.loads(itm)
                #print('lines: ', j['lines'])
                wiki_data[j['id']] = j['lines'].split('\n')
                #print(wiki_data[id])
                #print(wiki_data[j['id']])
                # for article in itm:
                #     wiki_data[article['id']] = read_text(article['lines'])
                print("Parsing {} of {} data chunks. Total entries: {}".format(i+1, len(json_files), len(wiki_data)))

    # create csv to write to
    with open('cleaned_wiki_data.csv', 'w+') as data:
        csv_data = csv.writer(data)
        csv_data.writerow(['id', 'verifiable', 'label', 'claim', 'evidence'])

        # collect evidence ids and pair to wiki_data
        evidence_dict = defaultdict(list)
        with open('train.jsonl') as jreader:
            for itm in jreader:
                j = json.loads(itm)
                #print("itm: ", j)
                id = str(j['id'])
                evidence_dict[id] = []
                for e in j['evidence']:
                    #print('evidence level 1: ', e)
                    for evidence in e:
                        anno_id = evidence[0]
                        evidence_id = evidence[1]
                        article_name = evidence[2]
                        sentence_id = evidence[3]
                        if sentence_id is not None:
                            #print(wiki_data[j['id']])
                            #print(wiki_data[j['id']][sentence_id])
                            try:
                                evidence_dict[id].append(wiki_data[article_name][sentence_id])
                                print("evidence_id:(", evidence_id, ")\narticle_name:(", article_name, ")\nsentence_id:(", sentence_id, ")")
                            except KeyError:
                                print(article_name, ' is not in available evidence.')
                                pass
                csv_data.writerow([j['id'], j['verifiable'], j['label'], j['claim'], evidence_dict[id]])

if __name__ == '__main__':
    main()
