from gensim.scripts.glove2word2vec import glove2word2vec
import argparse

# python -m gensim.scripts.glove2word2vec -i glove.6B.300d.txt -o glove.6B.300d.word2vec.txt

def convert(**hparams):
    print('[CONVERT] Starting to convert glove embeddings to gensim KeyedVectors encoding')
    glove2word2vec(glove_input_file=hparams['input_dir'], word2vec_output_file=hparams['output_dir'])
    print('[CONVERT] Finished!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to glove embedding file of any dimension',
        dest='input_dir')
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Path and filename of converted gensim KeyedVectors file',
        dest='output_dir')
    args = parser.parse_args()

    convert(**vars(args))