import json
import os
import csv
import numpy as np
from unicodedata import normalize
from collections import defaultdict
from random import shuffle

def main():
    # path to wiki-dump files
    path_to_data = r"wiki-pages/"
    # collect the names of all wiki files
    json_files = [fname for fname in os.listdir(path_to_data) if fname.startswith('wiki-') and fname.endswith('.jsonl')]

    def choice_from_list(some_list, sample_size):
        """
        Input: a list and desired sample size
        Returns: a random sample of the desired size, the same list with this sample removed
        """
        res = np.random.choice(some_list, sample_size, replace = False)
        new_list = [x for x in some_list if x not in res]
        return res, new_list

    # compile wiki files into one massive dictionary
    wiki_data = {}
    for i,f in enumerate(json_files):
        with open(os.path.join(path_to_data, f)) as jreader:
            for itm in jreader:
                j = json.loads(itm)
                wiki_data[normalize('NFC', j['id'])] = j['lines'].split('\n')
            print("Parsing {} of {} data chunks. Total entries: {}".format(i+1, len(json_files), len(wiki_data)))


    # create csv to write to (split into train, test, eval sets)
    with open('train_data_vanilla.csv', 'w+') as tr_data, open('test_data_vanilla.csv', 'w+') as te_data, open('eval_data_vanilla.csv', 'w+') as ev_data:
        train_data = csv.writer(tr_data)
        test_data = csv.writer(te_data)
        eval_data = csv.writer(ev_data)
        train_data.writerow(['id', 'verifiable', 'label', 'claim', 'evidence'])
        test_data.writerow(['id', 'verifiable', 'label', 'claim', 'evidence'])
        eval_data.writerow(['id', 'verifiable', 'label', 'claim', 'evidence'])

        # collect evidence ids and pair to wiki_data based on sentence index
        evidence_dict = defaultdict(dict)
        with open('train.jsonl') as jreader:
            for itm in jreader:
                j = json.loads(itm)
                id = str(j['id'])
                evidence_dict[id] = {}
                #dictionaries separating data into their categories (supports, refutes, nei)
                supports_dict = {}
                refutes_dict = {}
                nei_dict = {}
                #initialize each id's evidence as an empty list
                evidence_dict[id]['evidence'] = []
                for e in j['evidence']:
                    for evidence in e:
                        anno_id = evidence[0]
                        evidence_id = evidence[1]
                        article_name = evidence[2]
                        sentence_id = evidence[3]
                        if sentence_id is not None:
                            try:
                                article_name = normalize('NFC', article_name)
                                current_sentence = wiki_data[article_name][sentence_id].split('\t')[1]
                                if current_sentence not in evidence_dict[id]['evidence']:
                                    evidence_dict[id]['evidence'].append(current_sentence)
                                #print("evidence_id:(", evidence_id, ")\narticle_name:(", article_name, ")\nsentence_id:(", sentence_id, ")")
                            except KeyError as e:
                                print(article_name, ' is not in available evidence.')
                                pass
                # change 'verifiable' and 'label' into integers for easier manipulation
                if j['verifiable'] == 'VERIFIABLE':
                    verifiable = 1
                else:
                    verifiable = 0
                if j['label'] == 'SUPPORTS':
                    label = 1
                elif j['label'] == 'REFUTES':
                    label = 2
                else:
                    label = 0
                evidence_dict[id]['verifiable'] = verifiable
                evidence_dict[id]['label'] = label
                evidence_dict[id]['claim'] = j['claim']

        # sort data into supports/refutes/nei
        for key, data in evidence_dict.items():
            if data['label'] == 1:
                supports_dict[key] = data
            elif data['label'] == 2:
                refutes_dict[key] = data
            else:
                nei_dict[key] = data

        support_keys = list(supports_dict.keys())
        refute_keys = list(refutes_dict.keys())
        nei_keys = list(nei_dict.keys())
        #print(len(support_keys), len(refute_keys), len(nei_keys))

        # separate data into test, eval, and train (eval and test should each have 14500 data entries; train gets the rest)
        n = 14500 // 3
        test_support_keys, support_keys = choice_from_list(support_keys, n)
        eval_support_keys, train_support_keys = choice_from_list(support_keys, n)
        test_refute_keys, refute_keys = choice_from_list(refute_keys, n)
        eval_refute_keys, train_refute_keys = choice_from_list(refute_keys, n)
        test_nei_keys, nei_keys = choice_from_list(nei_keys, n)
        eval_nei_keys, train_nei_keys = choice_from_list(nei_keys, n)

        # add together support, refute, and nei keys, then shuffle them for randomization
        train_keys = np.concatenate((train_support_keys, train_refute_keys, train_nei_keys))
        eval_keys = np.concatenate((eval_support_keys, eval_refute_keys, eval_nei_keys))
        test_keys = np.concatenate((test_support_keys, test_refute_keys, test_nei_keys))
        shuffle(train_keys)
        shuffle(eval_keys)
        shuffle(test_keys)

        files = [(train_keys, train_data), (eval_keys, eval_data), (test_keys, test_data)]
        for keys, file in files:
            for k in keys:
                try:
                    file.writerow([k, evidence_dict[k]['verifiable'], evidence_dict[k]['label'], evidence_dict[k]['claim'], evidence_dict[k]['evidence']])
                except KeyError as e:
                    print(e)
                    raise

                #csv_data.writerow([int(id), verifiable, label, evidence_dict[id]['claim'], evidence_dict[id]['evidence_dict']])


if __name__ == '__main__':
    main()