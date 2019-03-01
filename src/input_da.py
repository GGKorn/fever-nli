import os
import sys
import yaml
import argparse
import unicodedata
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import iglob
from ast import literal_eval
from gensim.models.keyedvectors import KeyedVectors

TrainingPath = "train_data_*.csv"
EvalPath = "eval_data_*.csv"
TestPath = "test_data_*.csv"

# read embed path and dimensions from config file
with open('./config/config.yaml') as cnf:
    configs = yaml.load(cnf)
    try:
        W2V_PATH = configs['embed_path']
        EMBEDDING_SIZE = configs['embed_dims']
    except KeyError as e:
        print('Missing flags in the config file were found!')
        sys.exit(1)

# load embedding file once here, then later retrieve it to prevent garbage collection and re-loading that would increase
# runtime polynomially
global_embedding = KeyedVectors.load_word2vec_format(W2V_PATH, binary=False)

def get_input_fn_da(mode=None):
    """Creates an input function that loads the mode dependent data set and prepares it for use.

    Parameters:
        mode:   string, "train", "eval" or "predict": determines the data to be loaded
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

        # load embeddings
        vectors = global_embedding

        # specify "display" batch size for the output shapes of the iterators 
        batch_size = None
        cutoff_len = params.cutoff_len
        with tf.device('/cpu:0'):
            if mode == 'train':
                # dataset generator method takes the actual batch size because it relies on that for batch production
                ds_gen = get_dataset_generator(os.path.join(params.data_dir, TrainingPath), vectors, cutoff_len, params.batch_size)

                dataset = tf.data.Dataset.from_generator(
                    generator = ds_gen,
                    output_types = (tf.float32, tf.float32, tf.int64, tf.int64, tf.int64, tf.int64),
                    output_shapes = ([batch_size, cutoff_len, EMBEDDING_SIZE], [batch_size, cutoff_len, EMBEDDING_SIZE],
                                     [batch_size], [batch_size], [batch_size], [batch_size])
                )
                # shuffles and repeats the dataset. Relatively low shuffle buffer due to memory constraints, but randomness
                # should nevertheless be guaranteed
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10, count=None))
                # prefetching prepares another batch in case the CPU is idling during GPU computations
                dataset = dataset.prefetch(buffer_size=1)
            elif mode == 'eval':
                ds_gen = get_dataset_generator(os.path.join(params.data_dir, EvalPath), vectors, cutoff_len, params.batch_size)
                dataset = tf.data.Dataset.from_generator(
                    generator = ds_gen,
                    output_types = (tf.float32, tf.float32, tf.int64, tf.int64, tf.int64, tf.int64),
                    output_shapes = ([batch_size, cutoff_len, EMBEDDING_SIZE], [batch_size, cutoff_len, EMBEDDING_SIZE],
                                     [batch_size], [batch_size], [batch_size], [batch_size])
                )
                dataset = dataset.prefetch(buffer_size=1)
            elif mode == 'predict':
                ds_gen = get_dataset_generator(os.path.join(params.data_dir, TestPath), vectors, cutoff_len, params.batch_size)
                dataset = tf.data.Dataset.from_generator(
                    generator = ds_gen,
                    output_types = (tf.float32, tf.float32, tf.int64, tf.int64, tf.int64, tf.int64),
                    output_shapes = ([batch_size, cutoff_len, EMBEDDING_SIZE], [batch_size, cutoff_len, EMBEDDING_SIZE],
                                     [batch_size], [batch_size], [batch_size], [batch_size])
                )
                dataset = dataset.prefetch(buffer_size=1)
            else:
                raise ValueError('_input_fn received invalid MODE')

            # turn dataset into an iterator, set the functions return value as the first object in the iterator
            dataset = dataset.make_one_shot_iterator().get_next()
            # construct tuple out of the 6 different iterators, since tf.estimator expects 2 inputs: features and labels
            # the first 4 iterators (claims, evidence, evidence_length, claim_length) form the features, labels is formed
            # by (labels, verifiable_labels)
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
        Yields batches of claim data. Order: claims, evidences, evidence_length, claim_length, label, verifiablity
        Shapes in order: 2x (batch_size,embedding_size), 4x (batch_size,)
        """
        claims, evidences, labels, verify_labels = get_fever_claim_evidence_pairs(file)

        def lookup_emb(target):
            """
            Looks up the word embeddings for the whole sequence.
            Attention: large sequences are cut of past cutoff_len.
            Otherwise the data would blow up in size through padding.

            Parameters:
                target: string, the raw string

            Returns:
                 An array of shape (num_tokens,embedding_size)
            """
            embedding_list = []
            for i, token in enumerate(target.split()):
                # stop before cutoff length to prevent overshooting
                if i > cutoff_len-2:
                    break
                # normalise the strings prior to using them for lookups to remove unicode characters that would break
                # dictionary access
                token = unicodedata.normalize("NFD",token)
                if token in emb_vectors:
                    embedding_list.append(emb_vectors[token])
                else:
                    # use a random uniform vector for tokens not in the vocabulary
                    rand_embed = np.random.uniform(-0.5, 0.5, size=(EMBEDDING_SIZE))
                    embedding_list.append(rand_embed)
            return np.array(embedding_list).reshape(-1, EMBEDDING_SIZE)

        evid_max_len = cutoff_len - 1
        claim_max_len = cutoff_len - 1

        batch_start = 0
        # iteration steps are batch_size large. This leads to a cutoff effect at the end of a dataset iteration that drops
        # leftover samples below the specified batch size. Partly because Tensorflow cannot handle irregular batch sizes,
        # but also because smaller batches do not represent the dataset as well
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
                single_claim_emb = np.pad(single_claim_emb, ((1, claim_max_len - single_claim_emb.shape[0]), (0, 0)), mode='constant')
                single_evidence_emb = np.pad(single_evidence_emb, ((1, evid_max_len - single_evidence_emb.shape[0]), (0, 0)), mode='constant')

                # fill batches with embedded sentence
                emb_claims.append(single_claim_emb)
                emb_evidence.append(single_evidence_emb)

            # stack all the finished collections to turn them into numpy arrays for tensor-processing
            yield   np.stack(emb_claims), \
                    np.stack(emb_evidence), \
                    np.stack(emb_evidence_lens), \
                    np.stack(emb_claim_lens), \
                    np.stack(labels[batch_start:batch_end]), \
                    np.stack(verify_labels[batch_start:batch_end])
            batch_start = batch_end

    return _load_fever


def get_fever_claim_evidence_pairs(file_pattern,concat_evidence=True):
    """
    Reads the data files and returns the relevant data

    Parameters:
        file_pattern: pattern for finding potentially many files
        concat_evidence: bool, if all potential evidence strings are joined together. currently only True supported

    Returns:
         A tuple of 4 vectors, order: claim, evidence, label, verifiable_label
    """

    file_list = iglob(file_pattern)
    # converter reads in string in csv cell as list
    converter = {"evidence": literal_eval}
    # reads in all files that match the provided file pattern handle and concatenates them into one record
    data_frame = pd.concat(pd.read_csv(f, converters=converter) for f in file_list)

    concatenate = lambda x: " ".join(x)
    evidence_concat = data_frame["evidence"].apply(concatenate)
    if concat_evidence:
        evidence_list = list(evidence_concat)
    else:
        assert False, "No return of separate evidences supported"

    claim_list = list(data_frame["claim"])
    label_list = list(data_frame["label"])
    verif_list = list(data_frame["verifiable"])

    return claim_list, evidence_list, label_list, verif_list


if __name__ == "__main__":
    # just for testing or visualisation purpose

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--input_file", help="absolute path to the training/test file",required=True)
    parser.add_argument("-b", "--max_batches", help="maximum of batches outputed")
    parser.add_argument("-s", "--batch_size", help="size of batches outputed",default=64)
    parser.add_argument("-v", "--vocab_limit", help="maximum of word embeddings loaded from vocabulary")

    args = parser.parse_args()
    from timeit import default_timer as timer
    start = timer()


    if args.vocab_limit is not None:
        # loading fewer word vectors speeds up time till the actuall processing starts
        vectors = KeyedVectors.load_word2vec_format(W2V_PATH, binary=False,limit=args.vocab_limit)
    else:
        vectors = KeyedVectors.load_word2vec_format(W2V_PATH, binary=False)
    emb = timer()
    print('Loading embeddings took {} seconds'.format(emb - start))
    ds_gen = get_dataset_generator(args.input_file,vectors, args.batch_size)
    i = 0
    for a,b,c,d,e,f in ds_gen():
        i += 1
        print(i, ":", a.shape, b.shape, c.shape, d.shape, e.shape, f.shape)
        if args.max_batches and  i > args.max_batches:
            break

    print("iterations: ", i)
    end = timer()
    print("Preprocessing and batching took {} seconds".format(end - emb))