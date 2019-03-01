import tensorflow as tf

class DecomposibleAttentionModel(object):
    """
    Implementation of the model described in "A Decomposible Attention Model for Natural Language Inference", published
    by Parikh et al. in 2016.
    """
    def __init__(self, features, labels, mode, hparams):
        """
        Initialises the model with parameters passed from the estimator.
        
        Parameters:
            features:   model inputs, passed from the input_fn
            labels:     labels, passed from the input_fn
            mode:       tf.estimator.ModeKeys value [TRAIN, EVAL, PREDICT], denotes purpose of current run
            hparams:    command line arguments specifying hyperparameters
        """
        self.hparams        = hparams       # model hyperparameters, passed in through the command line
        self.mode           = mode          # current mode of execution
        self.evidence       = features[1]   # [?,  sentence_length, embedding_size], pre-embedded
        self.evidence_len   = features[2]   # [?], token length of evidence lists
        self.claims         = features[0]   # [?,  sentence_length, embedding_size], pre-embedded
        self.claims_len     = features[3]   # [?], token length of claim lists
        self.labels         = labels[0]     # [?], sparse labels
        self.verifiable     = labels[1]     # [?], whether or not ev-cl pair is verifiable (has enough info)
        self.global_step    = tf.Variable(0, trainable=False, name='global_step')

        # fixed hyperparameters that will not be available via commandline
        self.hidden_units = 200
        self.l2_lambda = 0.001
        self.clip_gradients = True
        self.clip_norm = 10
        self.dropout_keep_prob = 0.8
        self.logit_dims = 3

        self._build_model(mode)

    def _build_model(self, mode):
        """
        Builds the model and prepares return values for EstimatorSpecs.

        Parameters:
            mode:   tf.estimator.ModeKeys value [TRAIN, EVAL, PREDICT], denotes mode of execution of current run
        """
        with tf.variable_scope('model'):
            # Execute the 3-step approach outlined in the paper
            alpha, beta = self._attend()
            v_1i, v_2j = self._compare(alpha, beta)
            self.logits = self._aggregate(v_1i, v_2j)

            # specify the model's objective calculations
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
                logits_verifiable = tf.cast(tf.greater(predicted_classes, 0), tf.int64)
                self.eval_metric_ops = {
                    'accuracy': tf.metrics.accuracy(labels=self.labels, predictions=predicted_classes)
                    # malfunctioning in the current state
                    # 'true_pos': tf.metrics.true_positives(labels=self.verifiable, predictions=logits_verifiable),
                    # 'false_pos': tf.metrics.false_positives(labels=self.verifiable, predictions=logits_verifiable),
                    # 'true_neg': tf.metrics.true_negatives(labels=self.verifiable, predictions=logits_verifiable),
                    # 'false_neg': tf.metrics.false_negatives(labels=self.verifiable, predictions=logits_verifiable),
                    # 'recall': tf.metrics.true_positives(labels=self.verifiable, predictions=logits_verifiable),
                    # 'precision': tf.metrics.true_positives(labels=self.verifiable, predictions=logits_verifiable),
                    # 'f1': tf.contrib.metrics.f1_score(labels=self.verifiable, predictions=logits_verifiable)
                }

            with tf.variable_scope('optimisation'):
                # init optimiser, compute gradients, apply clipping during TRAIN
                optimizer = tf.train.AdagradOptimizer(learning_rate=self.hparams.learning_rate)
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                if self.clip_gradients:
                    gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
                self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())

    def _attend(self):
        """
        Paragraph 3.1 "Attend" from the paper.

        Returns:
            alpha and beta, soft-aligned vectors
        """
        with tf.variable_scope('attend'):
            # feed evidence and claims through first part of the network
            f_a = self._dense_layer(self.evidence,  self.hidden_units, 'attend_layers', reuse=False)
            f_b = self._dense_layer(self.claims,    self.hidden_units, 'attend_layers', reuse=True)

            # content of unnormalised attention weights e[i][j] denotes relation between i-th token of the evidence and 
            # j-th token of the claim
            unnorm_attn_weights = tf.matmul(f_a, f_b, transpose_b=True, name='unnorm_attn_weights')

            # produce masks of shape [1, 1, 1, 1, 0, 0, 0] to signify which values are padded (0) and which are real (1)
            # calculate logarithm of that mask to produce [0, 0, 0, 0, -inf, -inf, -inf]
            # adding -inf to the padded 0's will cancel them out in the softmax calculation
            # because e^(-inf [very small negative]) is basically 0
            masked_a = tf.add(tf.expand_dims(tf.sequence_mask(self.evidence_len, self.hparams.cutoff_len, dtype=tf.float32), -1), unnorm_attn_weights)
            masked_b = tf.add(tf.expand_dims(tf.sequence_mask(self.claims_len, self.hparams.cutoff_len, dtype=tf.float32), -1), tf.transpose(unnorm_attn_weights, [0, 2, 1]))

            # compute softmax on true values (sans padding)
            soft_attention_a = tf.nn.softmax(masked_a, name='soft_attention_a')
            soft_attention_b = tf.nn.softmax(masked_b, name='soft_attention_b')

            # finalised soft-alignment: tokens in evidence soft-aligned to tokens in claims, and vice-versa
            alpha = tf.matmul(soft_attention_b, self.evidence, name='alpha') # subphrase in a_bar that is softly aligned to b_bar
            beta = tf.matmul(soft_attention_a, self.claims, name='beta')     # subphrase in b_bar that is softly aligned to a_bar

            return alpha, beta

    def _compare(self, alpha, beta):
        """
        Paragraph 3.2 "Compare" from the paper.
        
        Parameters:
            alpha:  subphrase of claims that is soft-aligned to the evidence
            beta:   subphrase of evidence that is soft-aligned to the claims

        Returns:
            comparison-vectors v_1i (G([a_bar, beta])) and v_2j (G([b_bar, alpha]))
        """
        with tf.variable_scope('compare'):
            # concatenate embedding vector and corresponding alignment-vector
            a_bar_beta = tf.concat([self.evidence, beta], axis=2, name='a_beta_concat')
            b_bar_alpha = tf.concat([self.claims, alpha], axis=2, name='b_alpha_concat')

            # compute comparison
            v_1i = self._dense_layer(a_bar_beta, self.hidden_units, 'compare_layer', reuse=False)
            v_2j = self._dense_layer(b_bar_alpha, self.hidden_units, 'compare_layer', reuse=True)
            
            return v_1i, v_2j

    def _aggregate(self, v_1i, v_2j):
        """
        Paragraph 3.2 "Aggregate" from the paper.
        
        Parameters:
            v_1i:   [a_bar, beta]  comparison vector
            v_2j:   [b_bar, alpha] comparison vector

        Returns:
            Network logits, containing the output of the classifier        
        """
        with tf.variable_scope('aggregate'):
            v1 = tf.reduce_sum(v_1i, axis=1, name='v1_sum')
            v2 = tf.reduce_sum(v_2j, axis=1, name='v2_sum')
            v = tf.concat([v1,v2], axis=1)

            # predicted (unnormalised) scores for each class
            return self._dense_layer(v, self.hidden_units, 'aggregate_layer', reuse=False, final=True)

    def _dense_layer(self, input, size, scope = None, reuse = False, final=False):
        """
        Constructs the feed-forward blocks used in the paper. All layers will be relu-activated, He-initialised 
        (Delving Deep Into Rectifiers, He et al. 2015) and dropout-regularised. Re-use is applied because multiple
        values (evidence/claim, alpha/beta, v1/v2) are fed into the network per step of computation and not re-using
        would create a separate layer for each of those forward passes.

        Parameters:
            input:  layer input tensor
            size:   size of new layer
            scope:  optional scope for storing variables
            reuse:  True if TensorFlow should attempt to re-use implementation with identical scope name, false otherwise

        Returns:
            feed-forward dense layer of specified size with given inputs
        """
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope('layer1'):
                layer = tf.layers.dropout(input, (1 - self.dropout_keep_prob), training=(self.mode == tf.estimator.ModeKeys.TRAIN))
                layer = tf.layers.dense(layer,
                                        size,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

            with tf.variable_scope('layer2'):
                layer = tf.layers.dropout(layer, (1 - self.dropout_keep_prob), training=(self.mode == tf.estimator.ModeKeys.TRAIN))
                layer = tf.layers.dense(layer,
                                        size,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            if final:
                with tf.variable_scope('output'):
                    layer = tf.layers.dense(layer,
                                            self.logit_dims,
                                            activation=None,
                                            name='logits')
        return layer