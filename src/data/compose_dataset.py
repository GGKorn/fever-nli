import jsonlines
import os

# path to wiki-dump files
path_to_data = r"E:/Data/wikipedia_dump_2017/wiki-pages/"
# collect the names of all 
json_files = [fname for fname in os.listdir(path_to_data) if fname.startswith('wiki-') and fname.endswith('.jsonl')]

wiki_data = list()
for i,f in enumerate(json_files):
    with jsonlines.open(os.path.join(path_to_data, f)) as jreader:
        for itm in jreader:
            wiki_data.append(itm)
        print("Parsing {} of {} data chunks. Total entries: {}".format(i+1, len(json_files), len(wiki_data)))

with jsonlines.open('./data/train.jsonl') as jreader:
    for itm in jreader:
        print("itm: ", itm)
        for evidence in itm['evidence']:
            for _, eid, name, sid in evidence:
                # for every claim/label pair in train.jsonl, corresponding evidence sentence needs to be found in wiki_data
                # evidence name is the dict-key to the wiki page in wiki_data, util/misc.py::extract_sentence can be
                # used to extract extract sentences from the corpus
                print("evidence_id:(", eid, ") name:(", name, ") sentence_id:(", sid, ")") 
