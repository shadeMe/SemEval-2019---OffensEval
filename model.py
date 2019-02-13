from enum import Enum

import tensorflow as tf
from tensorflow.contrib import rnn
from common import *

class Phase(Enum):
	Train = 0
	Validation = 1
	Predict = 2


class Model:
	def create_char_embedding_layer(self, preproc, config, phase):
		# trainable character level embeddings
		char_embeddings = tf.get_variable(name="char_embeddings",
								   initializer=tf.random_uniform_initializer(-1.0, 1.0),
								   shape=[preproc.get_charvocab_size(), config.char_embedding_size],
								   trainable=phase == Phase.Train)

		return char_embeddings

	def create_word_embedding_layer(self, preproc, config, phase):
		# > trainable layer for the pretrained embeddings/base vocab
		# > trainable layer for unknown words that only occur in the training data
		# > non-trainable layer for unknown words that only occur in the validation data
		embedding_matrix = preproc.get_embeddings().get_data()
		embed_dim = preproc.get_embeddings().get_dim()
		embed_size = preproc.get_embeddings().get_size()

		#pretrained = tf.get_variable(
		#						name="embs_pretrained",
		#						shape=[embed_size, embed_dim],
		#						initializer=tf.constant_initializer(np.asarray(embedding_matrix),
		#						dtype=tf.float32),
		#						trainable=phase == Phase.Train)
		self._embeddings = pretrained = tf.placeholder(tf.float32, [embed_size, embed_dim], name="embs_pretrained")
	#	pretrained = tf.Variable(self._embeddings, trainable=phase == Phase.Train)
		train_only = tf.get_variable(
								name="embs_train_only",
								shape=[preproc.get_vocab_size_trainonly(), embed_dim],
								initializer=tf.random_uniform_initializer(-1.0, 1.0),
								trainable=phase == Phase.Train)
		val_only = tf.get_variable(
								name="embs_val_only",
								shape=[preproc.get_vocab_size_valonly(), embed_dim],
								initializer=tf.zeros_initializer(),
								trainable=False)

		combine = tf.concat([pretrained, train_only, val_only], axis=0)

		return combine

	def create_rnn_layer(self, input, output_sizes, output_dropout, state_dropout, seq_len, phase, name):
		forward_cells = [rnn.LSTMCell(
			size,
			True,
		    name=name + "_rnn-fw_" + str(idx)) for (idx, size) in enumerate(output_sizes)]

		if phase == Phase.Train:
			forward_cells = [rnn.DropoutWrapper(cell, output_keep_prob=output_dropout, state_keep_prob=state_dropout) for cell in forward_cells]

		backward_cells = [rnn.LSTMCell(
			size,
			True,
		    name=name + "_rnn-bw_" + str(idx)) for (idx, size) in enumerate(output_sizes)]

		if phase == Phase.Train:
			backward_cells = [rnn.DropoutWrapper(cell, output_keep_prob=output_dropout, state_keep_prob=state_dropout) for cell in backward_cells]

		_, fstate, bstate = rnn.stack_bidirectional_dynamic_rnn(forward_cells,
											backward_cells,
											input,
											dtype=tf.float32,
											sequence_length=seq_len)

		hidden_layer = tf.concat([fstate[-1][-1], bstate[-1][-1]], axis=1)

		return hidden_layer

	def add_hidden_layer(self, input, output_size, layer_prefix, phase, dropout=None, activation=None):
		w = tf.get_variable(layer_prefix + "_w",
					shape=[input.shape[1], output_size],
					initializer=tf.contrib.layers.xavier_initializer())

		b = tf.get_variable(layer_prefix + "_b",
					shape=[output_size],
					initializer=tf.ones_initializer())

		if activation != None:
			output = activation(tf.matmul(input, w) + b)
		else:
			output = tf.matmul(input, w) + b

		if dropout != None and phase == Phase.Train:
			output = tf.nn.dropout(output, dropout)

		return (output, w)


	def __init__(
			self,
			preproc,
			config,
			batch_words,
			batch_words_lens,
			batch_chars,
			batch_chars_lens,
			batch_labels,
			batch_docs,
			phase=Phase.Predict):
		label_size = batch_labels.shape[2]

		# The integer-encoded words. input_size is the (maximum) number of
		# time steps.
		self._x = tf.placeholder(tf.int32, shape=[batch_words.shape[1], batch_words.shape[2]])

		# This tensor provides the actual number of time steps for each instance.
		self._lens = tf.placeholder(tf.int32, shape=[batch_words.shape[1]])

		# This tensor provides the character IDs of the sentence
		self._char_rep = tf.placeholder(tf.int32, shape=[batch_chars.shape[1], batch_chars.shape[2]])

		# This tensor provides the actual number of char time steps for each instance
		self._char_rep_lens = tf.placeholder(tf.int32, shape=[batch_chars.shape[1]])

		# The label distribution.
		if phase != Phase.Predict:
			self._y = tf.placeholder(tf.float32, shape=[batch_chars.shape[1], label_size])

		# Document indices for tfidf/sentiment lookup
		self._docs = tf.placeholder(tf.int32, shape=[batch_docs.shape[1]])

		# Word and character embeddings
		self._embeddings = tf.placeholder(tf.float32, [None, None])	# dummy placeholder

		combined_embeddings = None

		word_embeddings = self.create_word_embedding_layer(preproc, config, phase)
		word_embeddings = tf.nn.embedding_lookup(word_embeddings, self._x)

		word_embeddings = self.create_rnn_layer(word_embeddings,
											config.word_rnn_sizes,
											config.word_rnn_output_dropout,
											config.word_rnn_state_dropout,
											self._lens,
											phase,
											"word_embed")

		combined_embeddings = word_embeddings

		if config.use_char_embeddings:
			char_embeddings = self.create_char_embedding_layer(preproc, config, phase)
			char_embeddings = tf.nn.embedding_lookup(char_embeddings, self._char_rep)

			char_embeddings = self.create_rnn_layer(char_embeddings,
												config.char_rnn_sizes,
												config.char_rnn_output_dropout,
												config.char_rnn_state_dropout,
												self._char_rep_lens,
												phase,
												"char_embed")


			combined_embeddings = tf.concat([combined_embeddings, char_embeddings], axis=1)

		# aux embeddings
		aux_embeddings = None
		if config.use_tfidf_vectors:
			tfidf_matrix = tf.get_variable(
								name="tfidf_matrix",
								shape=[preproc.get_tfidf().get_size(), preproc.get_tfidf().get_dim()],
								initializer=tf.constant_initializer(np.asarray(preproc.get_tfidf().get_data()),dtype=tf.float32),trainable=False)
			tfidf_vector = tf.gather(tfidf_matrix, self._docs)
			aux_embeddings = tfidf_vector

		if config.use_sentiment_vectors:
			sent_matrix = tf.get_variable(
								name="sentiment_matrix",
								shape=[preproc.get_sentiment().get_size(), preproc.get_sentiment().get_dim()],
								initializer=tf.constant_initializer(np.asarray(preproc.get_sentiment().get_data()),dtype=tf.float32),trainable=False)
			sent_vector = tf.gather(sent_matrix, self._docs)
			aux_embeddings = sent_vector if aux_embeddings == None else tf.concat([aux_embeddings, sent_vector], axis=1)


		if aux_embeddings != None:
			combined_embeddings = tf.concat([combined_embeddings, aux_embeddings], axis=1)


		if config.use_final_hidden_layer:
			final_hidden_layer, _ = self.add_hidden_layer(combined_embeddings,
												config.final_hidden_layer_size,
												"final_hidden_layer",
												phase,
												config.final_hidden_layer_dropout,
												tf.nn.tanh)

			final_logits, final_logit_weights = self.add_hidden_layer(final_hidden_layer,
																label_size,
																"final_logits",
																phase,
																None,
																tf.nn.tanh)
		else:
			final_logits, final_logit_weights = self.add_hidden_layer(combined_embeddings,
																label_size,
																"final_logits",
																phase,
																None,
																tf.nn.tanh)

		if phase == Phase.Train or Phase.Validation:
			if config.use_weighted_loss:
				dataset = preproc.get_training_set() if phase == Phase.Train else preproc.get_validation_set()
				class_weights = dataset.get_class_ratios(preproc.numberer_label, complement=True)
				weights = tf.reduce_sum(class_weights * self._y, axis=1)
				unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._y, logits=final_logits)
				losses = unweighted_losses * weights
			else:
				losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._y, logits=final_logits)

			if config.use_l2_regularization:
				losses = tf.reduce_mean(losses + config.l2_beta * tf.nn.l2_loss(final_logit_weights))
			else:
				losses = tf.reduce_mean(losses)


			self._loss = tf.reduce_sum(losses)

		if phase == Phase.Train:
			self._train_op = tf.train.AdamOptimizer(0.01).minimize(losses)
			self._probs = tf.nn.softmax(final_logits)

		if phase == Phase.Validation:
			# Labels for the gold data.
			self._gold_labels = tf.argmax(self.y, axis=1)

			# Predicted labels
			self._pred_labels = tf.argmax(final_logits, axis=1)

			correct = tf.equal(self._gold_labels, self._pred_labels)
			correct = tf.cast(correct, tf.float32)
			self._accuracy = tf.reduce_mean(correct)

	@property
	def accuracy(self):
		return self._accuracy

	@property
	def lens(self):
		return self._lens

	@property
	def loss(self):
		return self._loss

	@property
	def probs(self):
		return self._probs

	@property
	def train_op(self):
		return self._train_op

	@property
	def x(self):
		return self._x

	@property
	def y(self):
		return self._y

	@property
	def char_rep(self):
		return self._char_rep

	@property
	def char_rep_lens(self):
		return self._char_rep_lens

	@property
	def docs(self):
		return self._docs

	@property
	def embeddings(self):
		return self._embeddings

	@property
	def gold_labels(self):
		return self._gold_labels

	@property
	def pred_labels(self):
		return self._pred_labels
