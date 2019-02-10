import tensorflow as tf

class DecomposibleAttentionModel(object):
    def __init__(self, features, labels, mode, hparams):
        """
        Initialises the model with parameters passed from the estimator.
        
        Parameters:
            features:   model inputs, passed from the input_fn
            labels:     labels, passed from the input_fn
            mode:       tf.estimator.ModeKeys value [TRAIN, EVAL, PREDICT], denotes purpose of current run
            hparams:    command line arguments specifying hyperparameters
        """
        self.hparams = hparams
        self.evidence = features[0]
        self.claims = features[1]
        self.labels = labels
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # fixed hyperparameters that will not be available via commandline
        self.len_sentence = None
        self.vocab_size = None
        self.embedding_size = 300
        self.hidden_units = 200
        self.l2_lambda = 0.001
        self.clip_gradients = True
        self.clip_norm = 10
        self.dropout_keep_prob = 0.8

        self._build_model(mode)

    def _build_model(self, mode):
        """
        Builds the model and prepares return values for EstimatorSpecs.

        Parameters:
            mode:   tf.estimator.ModeKeys value [TRAIN, EVAL, PREDICT], denotes purpose of current run
        """
        with tf.variable_scope('model'):
            e_evidence, e_claims = self._embedding_lookup(self.evidence, self.claims)
            alpha, beta = self._attend(e_evidence, e_claims)
            v_1i, v_2j = self._compare(e_evidence, e_claims, alpha, beta)
            self.logits = self._aggregate(v_1i, v_2j)

            with tf.variable_scope('objective'):
                if mode == tf.estimator.ModeKeys.TRAIN:
                    # gather l2 coefficients and compute penalty
                    l2_coefficients = [tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'bias' not in var.name]
                    l2_penalty = tf.multiply(tf.add_n(l2_coefficients), self.l2_lambda)
                
                # calculate base xentropy loss, add l2 penalty in case of TRAIN
                xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels)
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
                    'accuracy': tf.metrics.accuracy(labels=tf.argmax(self.labels, 1), predictions=predicted_classes)
                }

            with tf.variable_scope('optimisation'):
                # init optimiser, compute gradients, apply clipping during TRAIN
                optimizer = tf.train.AdagradOptimizer(learning_rate=self.hparams.learning_rate)
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                if self.clip_gradients:
                    gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
                self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())

    def _embedding_lookup(self, evidence, claims):
        """
        Lookup embeddings from pre-trained word embedding matrix.
        
        Parameters:
            evidence:
            claims:

        Returns:
            tuple of embedding-vectors containing the information from evidence and claims
        """
        with tf.device('/cpu:0'):
            return evidence, claims

    def _attend(self, e_evidence, e_claims):
        """
        Paragraph 3.1 "Attend" from the paper.

        Parameters:
            e_claims:   embedding vector of the claims
            e_evidence: embedding vector of the evidence

        Returns:
            alpha and beta, soft-aligned vectors
        """
        with tf.variable_scope('attend'):
            f_a = self._dense_layer(e_evidence, self.hidden_units, 'attend_layers', reuse=False)
            f_b = self._dense_layer(e_claims, self.hidden_units, 'attend_layers', reuse=True)

            # content of unnormalised attention weights e[i][j] denotes relation between i-th token of the evidence and 
            # j-th token of the claim
            unnorm_attn_weights = tf.matmul(f_a, f_b, transpose_b=True, name='unnorm_attn_weights')
            soft_attention_a = tf.nn.softmax(unnorm_attn_weights, name='soft_attention_a')
            soft_attention_b = tf.nn.softmax(tf.transpose(unnorm_attn_weights, [0, 2, 1]), name='soft_attention_b')

            # finalised soft-alignment: tokens in evidence soft-aligned to tokens in claims, and vice-versa
            alpha = tf.matmul(soft_attention_b, e_evidence, name='alpha') # subphrase in a_bar that is softly aligned to b_bar
            beta = tf.matmul(soft_attention_a, e_claims, name='beta')     # subphrase in b_bar that is softly aligned to a_bar

            return alpha, beta

    def _compare(self, e_evidence, e_claims, alpha, beta):
        """
        Paragraph 3.2 "Compare" from the paper.
        
        Parameters:
            e_claims:   embedding vector of the claims
            e_evidence: embedding vector of the evidence
            alpha:      subphrase of claims that is soft-aligned to the evidence
            beta:       subphrase of evidence that is soft-aligned to the claims

        Returns:
            comparison-vectors v_1i (G([a_bar, beta])) and v_2j (G([b_bar, alpha]))
        """
        with tf.variable_scope('compare'):
            # concatenate embedding vector and corresponding alignment-vector
            a_bar_beta = tf.concat([e_evidence, beta], axis=2, name='a_beta_concat')
            b_bar_alpha = tf.concat([e_claims, alpha], axis=2, name='b_alpha_concat')

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
        Constructs the feed-forward blocks used in the paper. Optionally allows re-use to save memory. All layers will
        be relu-activated, He-initialised (Delving Deep Into Rectifiers, He et al. 2015) and dropout-regularised.

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
                layer = tf.nn.dropout(input, self.dropout_keep_prob)
                layer = tf.layers.dense(layer,
                                        size,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

            with tf.variable_scope('layer2'):
                layer = tf.nn.dropout(layer, self.dropout_keep_prob)
                layer = tf.layers.dense(layer,
                                        size,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            if final:
                with tf.variable_scope('output'):
                    layer = tf.layers.dense(layer,
                                            self.hparams.logit_dims,
                                            activation=None,
                                            name='logits')
        return layer