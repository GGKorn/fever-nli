import tensorflow as tf
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDFVec
from sklearn.feature_extraction.text import CountVectorizer as TFVec
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from ast import literal_eval


VERBOSE = True

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





def get_input_fn_nli(mode=None):
    """Creates an input function that loads the dataset and prepares it for use."""

    def _input_fn(mode=None, params=None):
        """
        Returns:
            An (one-shot) iterator containing (data, label) tuples
        """
        with tf.device('/cpu:0'):
            if mode == 'train':
                ds_gen = load_nli("example_train.")
            elif mode == 'eval' or mode == 'predict':
                ds_gen = load_nli("example_dev.csv")
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

def load_nli(file):
    """
    yields the data of next claim in order: claim, evidences, label
    :param file:
    :return:
    """


    claims,evidences,documents,labels = get_claim_evidence_pairs(file)

    tf_vec = TFVec(stop_words="english", max_features=5000).fit(documents)
    tfidf_vec = TFIDFVec(stop_words="english", max_features=5000).fit(documents)
    # TODO: store it, so can be called during testing
    del documents
    tf_claims = tf_vec.transform(claims)
    tf_evidences = tf_vec.transform(evidences)
    tfidf_claims = tfidf_vec.transform(claims)
    tfidf_evidences = tfidf_vec.transform(evidences)
    tfidf_sims = cosine_similarity(tfidf_claims,tfidf_evidences)
    del tfidf_claims, tfidf_evidences


    for tf_claim, tf_evidence,tfidf_sim,label  in zip(tf_claims,tf_evidences, tfidf_sims,labels):
        # see paper: p. 3 A_simple_but_tough_to_beat_baseline
        yield tf_claim, tfidf_sim, tf_evidence, label



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


def get_claim_evidence_pairs(file, concat_evidence=True):

    # converters: dict, optional
    #
    # Dict of functions for converting values in certain columns. Keys can either be integers or column labels.

    converter = {"evidence" : literal_eval}
    data_frame = pd.read_csv(file,converters=converter)


    concatenate = lambda x: " ".format(x)
    evidene_concat = data_frame["evidence"].apply(concatenate)
    document_list = evidene_concat + data_frame["claim"]
    if concat_evidence:
        evidence_list = evidene_concat
    else:
        evidence_list = list(data_frame["evidence"])

    claim_list = list(data_frame["claim"])
    label_list = list(data_frame["label"])


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
    a = load_nli(file="/workData/Uni/NLP/project/fever-nli/data/vanilla_wiki_data/train_data_vanilla.csv")
    print(a)