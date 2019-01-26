import argparse
import os
import tensorflow as tf

from input import get_input_fn

def get_model_fn():
    """Creates a model function that builds the net and manages estimator specs."""
    
    def _model_fn(features, labels, mode, params):
        """Builds the network model."""
        pass

    return _model_fn

def build_estimator(run_config, hparams):
    """Builds the estimator object and returns it."""
    return tf.estimator.Estimator(model_fn=get_model_fn(),
           config=run_config,
           params=hparams)

def main(**hparams):
    # Initialise TF logging
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info('Using arguments: {}\n'.format(str(hparams)))

    # Prepare ConfigProto object with several device settings
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_count={"CPU": hparams['num_cores'], "GPU": hparams['num_gpus']},
        gpu_options=tf.GPUOptions(force_gpu_compatible=True)
    )

    # for-loop for batch jobs
    for i in range(hparams['repeats']):
        tf.logging.info('Commencing iteration {} of {}.'.format((i+1), hparams['repeats']))

        config = tf.estimator.RunConfig(
            model_dir=os.path.join(hparams['output_dir'], '{}-{}'.format(str(hparams['job_id']), str(i+1))),
            tf_random_seed=None,
            save_summary_steps=100,
            save_checkpoints_steps=1000,
            save_checkpoints_secs=None,
            session_config=session_config,
            keep_checkpoint_max=int(hparams['train_steps']/1000)+1,
            keep_checkpoint_every_n_hours=10000,
            log_step_count_steps=100
        )

        classifier = build_estimator(
            run_config=config,
            hparams=tf.contrib.training.HParams(**hparams)
        )

        # start training and evaluation loop with estimator
        tf.estimator.train_and_evaluate(
            classifier,
            tf.estimator.TrainSpec(input_fn=get_input_fn(mode=tf.estimator.ModeKeys.TRAIN), max_steps=hparams['train_steps']),
            tf.estimator.EvalSpec(input_fn=get_input_fn(mode=tf.estimator.ModeKeys.EVAL), throttle_secs=1, steps=None)
        )

        tf.logging.info('Finished iteration {} of {}.\n'.format((i+1), hparams['repeats']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--data-dir',
        type=str,
        required=True,
        help='The directory where the input data is stored.',
        dest='data_dir')
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='The directory where the output will be stored.',
        dest='output_dir')
    parser.add_argument(
        '-j', '--job-id',
        type=int,
        required=True,
        help='The id this job was assigned during submission, alternatively any unique number distinguish runs.',
        dest='job_id')
    parser.add_argument(
        '-a', '--array-job',
        type=int,
        default=1,
        help='Number of scheduled repeats this job should run for.',
        dest='repeats')
    # parser.add_argument(
    #     '-d', '--logit-dimensions',
    #     type=int,
    #     required=True,
    #     help='Dimension of network logits',
    #     dest='logit_dims')
    parser.add_argument(
        '-n', '--num-gpus',
        type=int,
        default=1,
        help='The number of gpus used. Uses only CPU if set to 0.',
        dest='num_gpus')
    parser.add_argument(
        '-c', '--num-cpu-cores',
        type=int,
        default=1,
        help='The number of cpu cores available for data preparation.',
        dest='num_cores')
    parser.add_argument(
        '-s', '--train-steps',
        type=int,
        default=100,
        help='The number of steps to use for training.',
        dest='train_steps')
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=128,
        help='Batch size.',
        dest='batch_size')
    parser.add_argument(
        '-e', '--eval-batch-size',
        type=int,
        default=128,
        help="Evaluation batch size",
        dest="eval_batch_size")
    parser.add_argument(
        '-l', '--learning-rate',
        type=float,
        default=5e-4,
        help="""\
        This is the inital learning rate value.""",
        dest='learning_rate')
    args = parser.parse_args()

    main(**vars(args))
