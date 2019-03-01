import os
import unicodedata
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
from glob import iglob
from ast import literal_eval
from gensim.models.keyedvectors import KeyedVectors

debug = False
TrainingPath = "train_data_*.csv"
EvalPath = "eval_data_*.csv"
TestPath = "test_data_*.csv"

# W2V_PATH = r".\embedding\gensim_glove.6B.300d.txt"
W2V_PATH = r"E:\Python\ANLP Final Project\data\embedding\gensim_glove.6B.300d.txt"
emb_size_threshold = 500
emb_size = 300

# from https://stackoverflow.com/questions/48057991/get-word-embedding-dictionary-with-glove-python-model
# model.word_vectors[model.dictionary['samsung']]

def get_input_fn_da(mode=None):
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

        vectors = KeyedVectors.load_word2vec_format(W2V_PATH, binary=False)

        with tf.device('/cpu:0'):
            if mode == 'train':
                ds_gen = get_dataset_generator(os.path.join(params.data_dir, TrainingPath), vectors, params.cutoff_len, params.batch_size)

                dataset = tf.data.Dataset.from_generator(
                    generator = ds_gen,
                    output_types = (tf.float32, tf.float32, tf.int64, tf.int64, tf.int64, tf.int64)
                )
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10, count=None))
                dataset = dataset.prefetch(buffer_size=1)
            elif mode == 'eval':
                ds_gen = get_dataset_generator(os.path.join(params.data_dir, EvalPath), vectors, params.cutoff_len, params.eval_batch_size)
                dataset = tf.data.Dataset.from_generator(
                    generator = ds_gen,
                    output_types = (tf.float32, tf.float32, tf.int64, tf.int64, tf.int64, tf.int64)
                )
                # dataset = dataset.shuffle(buffer_size=1)
                dataset = dataset.prefetch(buffer_size=1)
            elif mode == 'predict':
                ds_gen = get_dataset_generator(os.path.join(params.data_dir, TestPath), vectors, params.cutoff_len, params.eval_batch_size)
                dataset = tf.data.Dataset.from_generator(
                    generator = ds_gen,
                    output_types = (tf.float32, tf.float32, tf.int64, tf.int64, tf.int64, tf.int64)
                )
                # dataset = dataset.shuffle(buffer_size=1)
                dataset = dataset.prefetch(buffer_size=1)
            else:
                raise ValueError('_input_fn received invalid MODE')

            dataset = dataset.make_one_shot_iterator().get_next()
            return (dataset[0], dataset[1], dataset[2], dataset[3]), (dataset[4], dataset[5])

    return _input_fn


def get_dataset_generator(file, emb_vectors, cutoff_len, batch_size=32):
    """
    Creates python generator that yields preprocessed, glove-embedded, padded and cut-off batches of the dataset.

    Parameters:
        file:           filename or name pattern to locate the input files
        emb_vectors:    word embedding vectors
        cutoff_len:     maximum token length after which an input sentence will be cut off
        batch_size:     number determining how many data samples get stacked together to form one batch

    Returns:
        A python generator that yields batches of the dataset.
    """

    def _load_fever():
        """
        yields the data of next claim in order: tf_claim, tfidf_sim, tf_evidence, label
        :param file:
        :return:
        """
        claims, evidences, labels, verify_labels = get_fever_claim_evidence_pairs(file)

        def lookup_emb(target):
            embedding_list = []
            for i, token in enumerate(target.split()):
                if i > cutoff_len:
                    break
                token = unicodedata.normalize("NFD",token)
                if token in emb_vectors:
                    embedding_list.append(emb_vectors[token])
                else:
                    rand_embed = np.random.uniform(size=(emb_size))
                    embedding_list.append(rand_embed)
            return np.array(embedding_list).reshape(-1, emb_size)

        evid_max_len = cutoff_len - 1
        claim_max_len = 60

        batch_start = 0
        for batch_end in range(batch_size, len(claims), batch_size):
            emb_claims, emb_evidence = [],[]
            emb_claim_lens, emb_evidence_lens = [], []
            for i in range(batch_start,batch_end,1):

                # get embdding for sentence
                single_claim_emb = lookup_emb(claims[i])
                single_evidence_emb = lookup_emb(evidences[i])

                # fill length vector for output
                emb_claim_lens.append(len(single_claim_emb))
                emb_evidence_lens.append(len(single_evidence_emb))

                # add zero-pad in front, also helps with empty evidences
                # zero-pad up to max_sent_length
                single_evidence_emb = np.pad(single_evidence_emb, ((1, evid_max_len - single_evidence_emb.shape[0]), (0, 0)), mode='constant')
                single_claim_emb = np.pad(single_claim_emb, ((1, claim_max_len - single_claim_emb.shape[0]), (0, 0)), mode='constant')

                # fill batches with embedded sentence
                emb_claims.append(single_claim_emb)
                emb_evidence.append(single_evidence_emb)

            # emb_evidence_lens, emb_claim_lens = np.array(emb_evidence_lens).reshape((batch_size,)), np.array(emb_claim_lens).reshape((batch_size,))
            yield   np.stack(emb_claims), \
                    np.stack(emb_evidence), \
                    np.stack(emb_evidence_lens), \
                    np.stack(emb_claim_lens), \
                    np.stack(labels[batch_start:batch_end]), \
                    np.stack(verify_labels[batch_start:batch_end])
            batch_start = batch_end

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
    from timeit import default_timer as timer
    start = timer()
    # local_test_path = "/workData/Uni/NLP/project/fever-nli/data/vanilla_wiki_data"
    local_test_path = r"E:\Python\ANLP Final Project\data\vanilla_wiki_data"
    if debug:
        vectors = KeyedVectors.load_word2vec_format(W2V_PATH, binary=False,limit=debug)
    else:
        vectors = KeyedVectors.load_word2vec_format(W2V_PATH, binary=False)
    emb = timer()
    print('Loading embeddings took {} seconds'.format(emb - start))
    ds_gen = get_dataset_generator(os.path.join(local_test_path,TestPath),vectors, 500)
    i = 0
    for a,b,c,d,e,f in ds_gen():
        i += 1
        print(i, ":", a.shape, b.shape, c.shape, d.shape, e.shape, f.shape)
        if i > 0:
            break
    print("iterations: ", i)
    end = timer()
    print("Preprocessing and batching took {} seconds".format(end - emb))
