import tensorflow as tf

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



def load_fever_chunk(file,chunk_size=100):
    """
    Fever specific data set loading by chunks
    :param file:
    :param chunk_size:
    :return:
    """
    #TODO: Treat special case: len(file) not deviable by chunk_size

    # not reading as csv as whole, so buffering pussible
    with open(file,"r")as fhandle:
        for line in fhandle:
            pass