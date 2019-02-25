import tensorflow as tf
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer as tf_vec

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
                ds_gen = load_nli("example_train.csv")
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
    Only yields verifiable
    yields the data of next claim in order: claim, evidences, label
    :param file:
    :return:
    """

    with open(file, "r") as read_file:
        data = json.load(read_file)

    #with open(file, "r") as json_file:
    #    print(json.load(json_file))
    #print("loaded {} data\ndata shape: {}".format(file, len(dict)))
    #print(dict)
    vectorizer = get_vectorizer(data)


    # for row in df:
    #     #TODO: see if verifiable is in NLI dataset
    #     if row["VERIFIABLE"]:
    #         # see paper: p. 3 A_simple_but_tough_to_beat_baseline
    #         yield None# tf_headline, tf_body, cosin-sim(tf_headline,tf_body)





def get_vectorizer(dict, mode="concat"):
    print("\nclaim\n", dict["claim"], "\n\nevidence\n", dict["evidence"])
    document_list = []
    for claim_set in dict:
        concat = claim_set["claim"]
        for evidence in claim_set["evidence"]:
            concat += evidence
        print(concat)
        document_list.append(concat)
    # claim_list = []
    # old_colname = None
    # for colname in ["claim","evidence"]:
    #     print(colname)
    #     if old_colname == None:
    #         concat = df[colname] + df[colname]
    #         print(concat)
    #         #" ".format([r for r in row])
    #    old_colname = colname




    return tf_vec.fit(document_list)

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
                ds_gen = load_fever_chunk("example_train.csv",chunk_size=2)
            elif mode == 'eval' or mode == 'predict':
                ds_gen = load_fever_chunk("example_dev.csv", chunk_size=2)
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

def load_fever_chunk(file,chunk_size=1000):
    """
    Fever specific data set loading by chunks
    :param file:
    :param chunk_size:
    :return:
    """
    #TODO: Treat special case: len(file) not dividable by chunk_size

    # pd.read_csv('pandas_dataframe_importing_csv/example.csv', na_values=sentinels, skiprows=3)
    # pd.read_csv('pandas_dataframe_importing_csv/example.csv', na_values=sentinels, skipfooter=2)

    for begin,end in []:
        df = pd.read_csv(file, skiprows=begin, skipfooter=end)




if __name__ == "__main__":
    a = load_nli(file="/workData/Uni/NLP/project/fever-nli/data/example_train.csv")
    print(a)