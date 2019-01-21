import jsonlines
import os
import pprint

# depending on number of sentence_id digits the extraction needs to begin at +1 cells to compensate
def extract_sentence(sentence_id, wiki_data, claim):
    if not wiki_data['id']:
        raise RuntimeError("extract_sentence() received call to read {} from empty wiki_data".format(sentence_id))
    sentence_start = wiki_data['lines'].find("{}\t".format(sentence_id)) + 2
    sentence_end = wiki_data['lines'].find(".\t", sentence_start) + 1
    print(sentence_start, sentence_end)
    if sentence_start == -1 or sentence_end == -1:
        raise RuntimeError("sentence_id {} not found for wiki_id {}".format(sentence_id, wiki_data['id']))
    else:
        return wiki_data['lines'][sentence_start:sentence_end]

path_to_data = r"E:/Data/wikipedia_dump_2017/wiki-pages/"
json_files = [fname for fname in os.listdir(path_to_data) if fname.endswith('.jsonl')]

wiki_data = list()
for i,f in enumerate(json_files):
    with jsonlines.open(os.path.join(path_to_data, f)) as jreader:
        for itm in jreader:
            wiki_data.append(itm)
        print("Parsing {} of {} data chunks. Total entries: {}".format(i+1, len(json_files), len(wiki_data)))
        break

pp = pprint.PrettyPrinter(indent=4)
# wiki_data[2]['testfield'] = "test"
# pp.pprint(wiki_data[2])
pp.pprint(extract_sentence(13, wiki_data[2], ""))

# with jsonlines.open('./data/train.jsonl') as jreader:
#     for itm in jreader:
#         print(itm)
#         break

# with open('./data/train.jsonl', 'r') as f:
#     res = [json.loads(jline) for jline in f]

# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(res[:10])