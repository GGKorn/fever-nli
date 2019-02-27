import tensorflow as tf
from ast import literal_eval
import os
import numpy as np
import scipy
from glob import iglob
import pandas as pd

debug = False
TrainingPath = "train_data_*.csv"
EvalPath = "eval_data_*.csv"
TestPath = "test_data_*.csv"


def get_input_fn(mode=None):
    """Creates an input function that loads the dataset and prepares it for use."""

    def _input_fn(mode=None, params=None):
        """
        Returns:
            An (one-shot) iterator containing (data, label) tuples
        """
        with tf.device('/cpu:0'):
            if mode == 'train':
                dataset = None
            elif mode == 'eval' or mode == 'predict':
                dataset = None
            else:
                raise ValueError('_input_fn received invalid MODE')
        return dataset

    return _input_fn


def get_dataset_generator(file=None, batch_size=500):
    def _load_fever():
        """
        yields the data of next claim in order: tf_claim, tfidf_sim, tf_evidence, label
        :param file:
        :return:
        """
        claims, evidences, documents, labels, verify_labels,  max_ev_len, max_cl_len = get_fever_claim_evidence_pairs(file)

        # print("fitting TF and tfidf")
        # tf_vec = TFVec(stop_words="english", max_features=5000).fit(documents)
        # tfidf_vec = TFIDFVec(stop_words="english", max_features=5000).fit(documents)
        # # print("finished fitting")
        # # TODO: store it, so can be called during testing
        # del documents
        # print("transform data")
        # tf_claims = tf_vec.transform(claims)
        # tf_evidences = tf_vec.transform(evidences)
        # tfidf_claims = tfidf_vec.transform(claims)
        # tfidf_evidences = tfidf_vec.transform(evidences)
        # # print("transformation done")
        # # print("tfidfs:\n claim: {}\n {},\n evidence: {}\n {}".format(tfidf_claims.shape,tfidf_claims,tfidf_evidences.shape,tfidf_evidences))
        #
        batch_start = 0
        for batch_end in range(500, len(claims), batch_size):

            yield (claims[batch_start:batch_end], evidences[batch_start:batch_end], max_ev_len, max_cl_len), (labels[batch_start:batch_end], verify_labels[batch_start:batch_end])
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
    max_ev_len = evidence_concat.map(lambda x: len(x)).max()
    max_cl_len = data_frame["claim"].map(lambda x: len(x)).max()
    document_list = list(evidence_concat + data_frame["claim"])
    if concat_evidence:
        evidence_list = list(evidence_concat)
    else:
        assert False, "No retrun of several evidences supported"
        #evidence_list = list(data_frame["evidence"])

    claim_list = list(data_frame["claim"])
    label_list = list(data_frame["label"])
    verif_list = list(data_frame["verifiable"])

    # print("loaded {} pairs".format(len(data_frame)))
    return claim_list, evidence_list, document_list, label_list, verif_list, max_ev_len, max_cl_len




if __name__ == "__main__":
    Embedding_Path = None
    local_test_path = "/workData/Uni/NLP/project/fever-nli/data/vanilla_wiki_data"
    # local_test_path = r"E:\Python\ANLP Final Project\data\vanilla_wiki_data"
    ds_gen = get_dataset_generator(os.path.join(local_test_path,TrainingPath))
    # print(ds_gen())
    i = 0
    for a,b in ds_gen():
        print(np.shape(a), np.shape(b))
        i += 1
    print("iterations: ", i)