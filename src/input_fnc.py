import os
import scipy
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import iglob
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer as TFVec
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDFVec

TrainingPath = "train_data_*.csv"
EvalPath = "eval_data_*.csv"
TestPath = "test_data_*.csv"

def get_input_fn_fnc(mode=None):
    """Creates an input function that loads the data set and prepares it for use.

    Parameters:
        mode: string, "train", "eval" or "predict": determines the data to be loaded

    """

    def _input_fn(mode=None, params=None):
        """
        Returns an (one-shot) iterator containing the batched data

        Parameters:
        mode:               string, "train", "eval" or "predict"; decides which data are loaded
        params.cutoff_len:  int, length past which strings are discarded
        params.data_dir:    str or path, directory where data for this mode lay
        params.batch_size:  int, the batch_size the generator yields in

        Returns:
            An (one-shot) iterator containing (data, label) tuples

        """
        with tf.device('/cpu:0'):
            if mode == 'train':
                # fixed output shapes because the output will always be batches of 
                # ([5000-dim TF vector of claims], [tf-idf cosine similary], [5000-dim TF vector of evidence])
                ds_gen = get_dataset_generator(os.path.join(params.data_dir, TrainingPath), params.batch_size)
                dataset = tf.data.Dataset.from_generator(
                    generator = ds_gen,
                    output_types = (tf.float32, tf.int64),
                    output_shapes = ([params.batch_size, 10001],[params.batch_size])
                )
                # shuffles and repeats the dataset. Relatively low shuffle buffer due to memory constraints, but randomness
                # should nevertheless be guaranteed
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10, count=None))
                # prefetching prepares another batch in case the CPU is idling during GPU computations
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
        # turn dataset into an iterator, set the functions return value as the first object in the iterator
        return dataset.make_one_shot_iterator().get_next()

    return _input_fn


def get_dataset_generator(file=None, batch_size=500):
    """
        Creates python generator that yields preprocessed batches of the dataset for the NLI model.
        Uses claim TF, evidence TF and TFIDF cosine similarity

        Parameters:
            file:           filename or name pattern to locate the input files
            batch_size:     number determining how many data samples get stacked together to form one batch

        Returns:
            A python generator that yields batches of the data set.
        """

    def _load_nli():
        """
        Yields a batch of the data of next claim in order: (tf_claim, tfidf_sim, tf_evidence), label
        """
        claims,evidences,documents,labels = get_claim_evidence_pairs(file)

        # collect TF information and tf-idf vectors, garbage-collect the documents to conserve memory
        tf_vec = TFVec(stop_words="english", max_features=5000).fit(documents)
        tfidf_vec = TFIDFVec(stop_words="english", max_features=5000).fit(documents)
        del documents
        # compute tf and tf-idf vectors for evidence and claims
        tf_claims = tf_vec.transform(claims)
        tf_evidences = tf_vec.transform(evidences)
        tfidf_claims = tfidf_vec.transform(claims)
        tfidf_evidences = tfidf_vec.transform(evidences)

        batch_start = 0
        for batch_end in range(batch_size, len(claims), batch_size):

            # compute cosine similary for tf-idf vectors. Only interesting in diagonal dimension, however sklearn's implementation
            # is much more effective/faster/cheap than manually computing only the diagonal
            tfidf_sims = np.diag(cosine_similarity(tfidf_claims[batch_start:batch_end,], tfidf_evidences[batch_start:batch_end,])).reshape(-1,1)

            yield scipy.sparse.hstack((tf_claims[batch_start:batch_end,], tfidf_sims, tf_evidences[batch_start:batch_end,])).A , labels[batch_start:batch_end]
            batch_start = batch_end

    return _load_nli


def get_claim_evidence_pairs(file_pattern, concat_evidence=True):
    """
        Reads the data files and returns the relevant data

        Parameters:
            file_pattern: pattern for finding potentially many files
            concat_evidence: bool, if all potential evidence strings are joined together. currently only True supported

        Returns:
             A tuple of 4 vectors, order: claims, evidences, documents, labels
    """

    converter = {"evidence" : literal_eval}
    file_list = iglob(file_pattern)
    # reads in all files that match the provided file pattern handle and concatenates them into one record
    data_frame = pd.concat(pd.read_csv(f, converters=converter) for f in file_list)

    # concatenate evidence string for documents anyway
    # documents are just used for training the vocabulary
    concatenate = lambda x: " ".join(x)
    evidence_concat = data_frame["evidence"].apply(concatenate)
    document_list = list(evidence_concat + data_frame["claim"])

    if concat_evidence:
        evidence_list = list(evidence_concat)
    else:
        assert False, "No return of separate evidences supported"

    claim_list = list(data_frame["claim"])
    label_list = list(data_frame["label"])

    return claim_list, evidence_list, document_list, label_list


if __name__ == "__main__":
    # just for testing or visualisation purpose
    pass