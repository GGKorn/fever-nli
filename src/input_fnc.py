import tensorflow as tf
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDFVec
from sklearn.feature_extraction.text import CountVectorizer as TFVec
from sklearn.metrics.pairwise import cosine_similarity
# import gensim
from ast import literal_eval
import os
import scipy
from glob import iglob


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
                ds_gen = get_dataset_generator(os.path.join(params.data_dir, TrainingPath), params.batch_size)
                dataset = tf.data.Dataset.from_generator(
                    generator = ds_gen,
                    output_types = (tf.float32, tf.int64),
                    output_shapes = ([params.batch_size, 10001],[params.batch_size])
                )
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10, count=None))
                dataset = dataset.prefetch(buffer_size=1)
            elif mode == 'eval':
                ds_gen = get_dataset_generator(os.path.join(params.data_dir, EvalPath), params.eval_batch_size)
                dataset = tf.data.Dataset.from_generator(
                    generator = ds_gen,
                    output_types = (tf.float32, tf.int64),
                    output_shapes = ([params.eval_batch_size, 10001],[params.eval_batch_size])
                )
                dataset = dataset.shuffle(buffer_size=1)
                dataset = dataset.prefetch(buffer_size=1)
            elif mode == 'predict':
                ds_gen = get_dataset_generator(os.path.join(params.data_dir, TestPath), params.eval_batch_size)
                dataset = tf.data.Dataset.from_generator(
                    generator = ds_gen,
                    output_types = (tf.float32, tf.int64),
                    output_shapes = ([params.eval_batch_size, 10001],[params.eval_batch_size])
                )
                dataset = dataset.shuffle(buffer_size=1)
                dataset = dataset.prefetch(buffer_size=1)
            else:
                raise ValueError('_input_fn received invalid MODE')
        return dataset.make_one_shot_iterator().get_next()

    return _input_fn


def get_dataset_generator(file=None, batch_size=500):

    def _load_nli():
        """
        yields the data of next claim in order: tf_claim, tfidf_sim, tf_evidence, label
        :param file:
        :return:
        """
        claims,evidences,documents,labels = get_claim_evidence_pairs(file)

        # print("fitting TF and tfidf")
        tf_vec = TFVec(stop_words="english", max_features=5000).fit(documents)
        tfidf_vec = TFIDFVec(stop_words="english", max_features=5000).fit(documents)
        # print("finished fitting")
        # TODO: store it, so can be called during testing
        del documents
        # print("transform data")
        tf_claims = tf_vec.transform(claims)
        tf_evidences = tf_vec.transform(evidences)
        tfidf_claims = tfidf_vec.transform(claims)
        tfidf_evidences = tfidf_vec.transform(evidences)
        # print("transformation done")
        #print("tfidfs:\n claim: {}\n {},\n evidence: {}\n {}".format(tfidf_claims.shape,tfidf_claims,tfidf_evidences.shape,tfidf_evidences))

        batch_start = 0
        for batch_end in range(500, len(claims), batch_size):
            # see paper: p. 3 A_simple_but_tough_to_beat_baseline


            # TODO: find how to get diagonal out of  n_samples_X, n_samples_Y matrix
            #  cosine_similarity returns An array with shape (n_samples_X, n_samples_Y).
            tfidf_sims = np.diag(cosine_similarity(tfidf_claims[batch_start:batch_end,], tfidf_evidences[batch_start:batch_end,])).reshape(-1,1)
            # tfidf_sims = np.array([cosine_similarity(x, y) for x, y in zip(tfidf_claims[batch_start:batch_end,], tfidf_evidences[batch_start:batch_end,])]).reshape(-1, 1)

            # TODO: check if concatenation works
            #print(tf_claims[batch_start:batch_end,].shape,tfidf_sims.shape,tf_evidences[batch_start:batch_end,].shape)

            yield scipy.sparse.hstack((tf_claims[batch_start:batch_end,], tfidf_sims, tf_evidences[batch_start:batch_end,])).A , labels[batch_start:batch_end]
            batch_start = batch_end

    return _load_nli


def load_fever(file,wordemb_path,concat_evidence=True):
    print("Loading pre-trained embedding", wordemb_path)

    with open(file, "r") as read_file:
        data = json.load(read_file)
    claims, evidences, documents, labels = get_claim_evidence_pairs(data,concat_evidence)

    # deprecated: gensim.models.Word2Vec.load_word2vec_format(wordemb_path, binary=True)
    vectors = gensim.models.Word2Vec.load_word2vec_format(wordemb_path, binary=True)

    we_claims, we_evidences = [],[]
    longest_evidence = None
    for we_claim, we_evidence, label  in zip(we_claims,we_evidences, labels):
        # see paper: p. 3 A_simple_but_tough_to_beat_baseline
        yield we_claim , we_evidence, label, longest_evidence


def get_claim_evidence_pairs(file_pattern, concat_evidence=True):

    # converters: dict, optional
    #
    # Dict of functions for converting values in certain columns. Keys can either be integers or column labels.
    # print("load claim-evidence pairs from {}".format(file))
    converter = {"evidence" : literal_eval}
    file_list = iglob(file_pattern)
    data_frame = pd.concat(pd.read_csv(f, converters=converter) for f in file_list)

    concatenate = lambda x: " ".join(x)
    evidence_concat = data_frame["evidence"].apply(concatenate)
    document_list = list(evidence_concat + data_frame["claim"])
    if concat_evidence:
        evidence_list = list(evidence_concat)
    else:
        evidence_list = list(data_frame["evidence"])

    claim_list = list(data_frame["claim"])
    label_list = list(data_frame["label"])

    # print("loaded {} pairs".format(len(data_frame)))
    return claim_list, evidence_list, document_list, label_list


#TODO: how to integrate word embedding transform
def get_input_fn_fever(mode=None):
    """Creates an input function that loads the dataset and prepares it for use."""

    def _input_fn(mode=None, params=None):
        """
        Returns:
            An (one-shot) iterator containing (data, label) tuples
        """
        with tf.device('/cpu:0'):
            if mode == 'train':
                ds_gen = load_fever("example_train.csv",Embedding_Path)
            elif mode == 'eval' or mode == 'predict':
                ds_gen = load_fever("example_dev.csv",Embedding_Path)
            else:
                raise ValueError('_input_fn received invalid MODE')

            #TODO: lookup tf.<data.type> and shape for these textdata/ask Ben
            dataset = tf.data.Dataset.from_generator(
                generator = ds_gen,
                output_types = (tf.int64, tf.int64),
                output_shapes = (tf.TensorShape([]), tf.TensorShape([None]))
            )

        return dataset.make_one_shot_iterator().get_next()

    return _input_fn
    

if __name__ == "__main__":
    Embedding_Path = None
    # local_test_path = "/workData/Uni/NLP/project/fever-nli/data/vanilla_wiki_data"
    # local_test_path = r"E:\Python\ANLP Final Project\data\vanilla_wiki_data"
    # ds_gen = get_dataset_generator(os.path.join(local_test_path,TrainingPath))
    # # print(ds_gen())
    # i = 0
    # for a,b in ds_gen():
    #     print(np.shape(a), np.shape(b))
    #     i += 1
    # print("iterations: ", i)