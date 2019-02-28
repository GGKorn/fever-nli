import tensorflow as tf
from ast import literal_eval
import os
import numpy as np
import scipy
from glob import iglob
import pandas as pd
import unicodedata


debug = False
TrainingPath = "train_data_*.csv"
EvalPath = "eval_data_*.csv"
TestPath = "test_data_*.csv"


from gensim.models.keyedvectors import KeyedVectors
W2V_PATH = "/workData/generalUseData/GoogleNews-vectors-negative300.bin.gz"
W2V_IS_BINARY = True
emb_size = 300



# from https://stackoverflow.com/questions/48057991/get-word-embedding-dictionary-with-glove-python-model
# model.word_vectors[model.dictionary['samsung']]



def get_input_fn(mode=None):
    """Creates an input function that loads the dataset and prepares it for use."""

    def _input_fn(mode=None, params=None):
        """
        Returns:
            An (one-shot) iterator containing (data, label) tuples
        """

        # keep
        # do processing here
        #   1. load WE
        #   2. load data
        #   3. get indexes of data from WE
        #   4. save index representaion locally
        #   5. pass file name to load_fever
        #   6. load the already indexed data in load_fever

        vectors = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)



        with tf.device('/cpu:0'):
            if mode == 'train':
                ds_gen = get_dataset_generator(os.path.join(params.data_dir, TrainingPath), vectors, params.batch_size)

                dataset = tf.data.Dataset.from_generator(
                    generator = ds_gen,
                    output_types = (tf.float32,tf.float32, tf.int64,tf.int64,tf.int64,tf.int64)
                )
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10, count=None))
                dataset = dataset.prefetch(buffer_size=1)
            elif mode == 'eval':
                ds_gen = get_dataset_generator(os.path.join(params.data_dir, EvalPath), vectors, params.eval_batch_size)
                dataset = tf.data.Dataset.from_generator(
                    generator = ds_gen,
                    output_types = (tf.float32,tf.float32, tf.int64,tf.int64,tf.int64,tf.int64)
                )
                dataset = dataset.shuffle(buffer_size=1)
                dataset = dataset.prefetch(buffer_size=1)
            elif mode == 'predict':
                ds_gen = get_dataset_generator(os.path.join(params.data_dir, TestPath), vectors, params.eval_batch_size)
                dataset = tf.data.Dataset.from_generator(
                    generator = ds_gen,
                    output_types = (tf.float32,tf.float32, tf.int64,tf.int64,tf.int64,tf.int64)
                )
                dataset = dataset.shuffle(buffer_size=1)
                dataset = dataset.prefetch(buffer_size=1)
            else:
                raise ValueError('_input_fn received invalid MODE')
        return dataset.make_one_shot_iterator().get_next()

    return _input_fn


def get_dataset_generator(file, emb_vectors, batch_size=500):



    def _load_fever():
        """
        yields the data of next claim in order: tf_claim, tfidf_sim, tf_evidence, label
        :param file:
        :return:
        """

        claims, evidences, labels, verify_labels = get_fever_claim_evidence_pairs(file)
        # to extract word vector
        #lookup_em = lambda x: [emb_vectors[token] for token in x.split() if token in emb_vectors]

        def lookup_em(x):

            embedding_list = []
            for token in x.split():
                token = unicodedata.normalize("NFD",token)
                if token in emb_vectors:
                    embedding_list.append(emb_vectors[token])
                else:
                    embedding_list.append(np.random.uniform(size=(emb_size)))
            return embedding_list

        # evid_max_len = map(lookup_em,evidences)
        evid_max_len = 0
        max_len_index = 0
        for i,sent in enumerate(evidences):
            sent_len = len(lookup_em(sent))
            if sent_len > evid_max_len:
                evid_max_len = sent_len
                max_len_index = i
        else:
            print("longest evi",evid_max_len,unicodedata.normalize("NFD",evidences[max_len_index]))
        #claim_max_len = map(lookup_em,claims)
        claim_max_len = 0
        max_len_index = 0
        for i, sent in enumerate(claims):
            sent_len = len(lookup_em(sent))
            if sent_len > claim_max_len:
                claim_max_len = sent_len
                max_len_index = i
        else:
            print("longest claim",claim_max_len,claims[max_len_index])


        for i in range(len(claims)):
            emb_claims = lookup_em(claims[i])
            emb_claim_len = len(emb_claims)
            emb_claims = emb_claims + [] * (claim_max_len - emb_claim_len)
            emb_eviden = lookup_em(evidences[i])
            emb_eviden_len = len(emb_eviden)



            if emb_eviden_len < 1 or emb_claim_len < 1:
                pass
            else:
                # TODO: (500,)
                yield emb_claims, emb_eviden, emb_eviden_len, emb_claim_len, labels[i], verify_labels[i]

    return _load_fever


def get_fever_claim_evidence_pairs(file_pattern,concat_evidence=True):


    # converters: dict, optional
    #
    # Dict of functions for converting values in certain columns. Keys can either be integers or column labels.
    # print("load claim-evidence pairs from {}".format(file))
    converter = {"evidence": literal_eval}
    file_list = iglob(file_pattern)
    data_frame = pd.concat(pd.read_csv(f, converters=converter) for f in file_list)

    concatenate = lambda x: " ".join(x)
    evidence_concat = data_frame["evidence"].apply(concatenate)
    # TODO: check if specifing col is required
    #ev_len = evidence_concat.map(lambda x: len(x))
    #max_ev_len = evidence_concat.map(lambda x: len(x.split())).max()
    #cl_len = data_frame["claim"].map(lambda x: len(x))
    #max_cl_len = data_frame["claim"].map(lambda x: len(x.split())).max()
    if concat_evidence:
        evidence_list = list(evidence_concat)
    else:
        assert False, "No retrun of several evidences supported"
        #evidence_list = list(data_frame["evidence"])

    claim_list = list(data_frame["claim"])
    label_list = list(data_frame["label"])
    verif_list = list(data_frame["verifiable"])

    # print("loaded {} pairs".format(len(data_frame)))
    return claim_list, evidence_list, label_list, verif_list




if __name__ == "__main__":
    Embedding_Path = None
    local_test_path = "/workData/Uni/NLP/project/fever-nli/data/vanilla_wiki_data"
    vectors = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)
    # local_test_path = r"E:\Python\ANLP Final Project\data\vanilla_wiki_data"
    ds_gen = get_dataset_generator(os.path.join(local_test_path,TrainingPath),vectors)
    # print(ds_gen())
    i = 0
    for a,b,c,d,e,f in ds_gen():
        print(np.shape(a), np.shape(b),e,f)
        i += 1
        if i > 10:
            break
    print("iterations: ", i)