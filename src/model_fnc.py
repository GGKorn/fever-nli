import tensorflow as tf

class SimpleBaselineModel(object):
    """
    Implementation of single hidden-layer FFN model of the FEVER baseline.
    """
    def __init__(self, features, labels, mode, hparams):
        """Initialises the model with all components."""
        self.hparams = hparams
        # fixed hyperparameters that will not be available via commandline
        self.hidden_units = 100
        self.l2_lambda = 0.00001
        self.clip_gradients = True
        self.clip_norm = 5
        self.dropout_keep_prob = 0.6
        self.logit_dims = 3

        with tf.name_scope('inputs'):
            self.inputs = features
            self.labels = labels
            self.learning_rate = hparams.learning_rate
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self._build_model(mode)

    def _build_model(self, mode):
        """
        Builds the model and prepares return values for EstimatorSpecs. Single-layer feed-forward network, relu-activated,
        He-initialised (Delving Deep Into Rectifiers, He et al. 2015) and dropout-regularised.

        Parameters:
            mode:   tf.estimator.ModeKeys value [TRAIN, EVAL, PREDICT], denotes purpose of current run
        """
        isTraining = (mode == tf.estimator.ModeKeys.TRAIN)
        with tf.variable_scope('model'):
            with tf.variable_scope('hidden_layer'):
                # hidden layer
                model = tf.layers.dense(self.inputs, 
                                        self.hidden_units,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                        name='dense01')
                model = tf.layers.dropout(model, (1 - self.dropout_keep_prob), training=isTraining)
            
            with tf.variable_scope('output_layer'):
                # output layer
                model = tf.layers.dense(model,
                                        self.logit_dims,
                                        activation=None,
                                        name='logits')
                self.logits = tf.reshape(model, [self.hparams.batch_size, self.logit_dims])

            # objective ops
            with tf.variable_scope('objective'):
                if mode == tf.estimator.ModeKeys.TRAIN:
                    # gather l2 coefficients and compute penalty
                    l2_coefficients = [tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'bias' not in var.name]
                    l2_penalty = tf.multiply(tf.add_n(l2_coefficients), self.l2_lambda)

                # calculate base xentropy loss, add l2 penalty in case of TRAIN
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
                xentropy = xentropy + l2_penalty if mode == tf.estimator.ModeKeys.TRAIN else xentropy
                self.loss = tf.reduce_mean(xentropy, name='loss_xentropy')

                # PREDICT EstimatorSpec
                predicted_classes = tf.argmax(self.logits, 1)
                self.predictions = {
                    'class': predicted_classes,
                    'prob': tf.nn.softmax(self.logits)
                }

                # EVAL EstimatorSpec
                self.eval_metric_ops = {
                    'accuracy': tf.metrics.accuracy(
                        labels=self.labels, predictions=predicted_classes)
                }

            with tf.variable_scope('optimisation'):
                # init optimiser, compute gradients, apply clipping during TRAIN
                optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams.learning_rate)
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                if self.clip_gradients:
                    gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
                self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())
