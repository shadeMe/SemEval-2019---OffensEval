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
								   shape=(preproc.get_charvocab_size(), config.char_embedding_size),
								   trainable=phase == Phase.Train)

		return char_embeddings

	def create_word_embedding_layer(self, preproc, config, phase):
		# > trainable layer for the pretrained embeddings/base vocab
		# > trainable layer for unknown words that only occur in the training data
		# > non-trainable layer for unknown words that only occur in the validation data
		embedding_matrix = preproc.get_embeddings().get_data()
		embed_dim = preproc.get_embeddings().get_dim()
		embed_size = preproc.get_embeddings().get_size()

		pretrained = tf.get_variable(
								name="embs_pretrained",
								shape=[embed_size, embed_dim],
								initializer=tf.constant_initializer(np.asarray(embedding_matrix), dtype=tf.float32),
								trainable=phase == Phase.Train)
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


	def __init__(
			self,
			preproc,
			config,
			batch,
			lens_batch,
			label_batch,
			word_lens,
			words,
			phase=Phase.Predict):
		batch_size = batch.shape[1]
		input_size = batch.shape[2]
		label_size = label_batch.shape[2]
		words_size = words.shape[3]

		# The integer-encoded words. input_size is the (maximum) number of
		# time steps.
		self._x = tf.placeholder(tf.int32, shape=[batch_size, input_size])

		# This tensor provides the actual number of time steps for each
		# instance.
		self._lens = tf.placeholder(tf.int32, shape=[batch_size])

		# This tensor provides the character IDs of words present
		self._words = tf.placeholder(tf.int32, shape=[batch_size, input_size, words_size])

		# This tensor provides the actual number of char time steps for each word
		self._word_lens = tf.placeholder(tf.int32, shape=[batch_size, input_size])

		# The label distribution.
		if phase != Phase.Predict:
			self._y = tf.placeholder(tf.float32, shape=[batch_size, label_size])

		# Word and summed-up character embeddings
		word_embeddings = self.create_word_embedding_layer(preproc, config, phase)
		word_embeddings = tf.nn.embedding_lookup(word_embeddings, self._x)

		#char_embeddings = self.create_char_embedding_layer(preproc, config, phase)
		#char_embeddings = tf.nn.embedding_lookup(char_embeddings, self._words)
		#char_embeddings = tf.reduce_sum(char_embeddings, 2)
		#combined_embeddings = tf.concat([word_embeddings, char_embeddings], 2)
		combined_embeddings = word_embeddings

		forward_cell = rnn.GRUCell(config.forward_cell_units)
		reg_forward_cell = rnn.DropoutWrapper(forward_cell, output_keep_prob=config.hidden_dropout)
		backward_cell = rnn.GRUCell(config.backward_cell_units)
		reg_backward_cell = rnn.DropoutWrapper(backward_cell, output_keep_prob=config.hidden_dropout)
		_, lstm_out_split = tf.nn.bidirectional_dynamic_rnn(reg_forward_cell,
															reg_backward_cell,
															combined_embeddings,
															sequence_length=self._lens,
															dtype=tf.float32)
		lstm_out = tf.concat(lstm_out_split, 1)

		w = tf.get_variable('weights', shape=[lstm_out.shape[1], label_size])
		b = tf.get_variable('bias', shape=[1])

		logits = tf.matmul(lstm_out, w) + b

		if phase == Phase.Train or Phase.Validation:
			losses = tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=logits)
			losses = tf.reduce_mean(losses + config.l2_beta * tf.nn.l2_loss(w))
			self._loss = loss = tf.reduce_sum(losses)

		if phase == Phase.Train:
			start_lr = 0.001
			global_step = tf.Variable(0, trainable=False)
			num_batches = preproc.get_training_set().get_size() / batch_size
			learning_rate = tf.train.exponential_decay(start_lr, global_step, num_batches, 0.9)
			self._train_op = tf.train.RMSPropOptimizer(start_lr, decay=0.8).minimize(losses)
			self._probs = probs = tf.nn.softmax(logits)

		if phase == Phase.Validation:
			# Labels for the gold data.
			gold_labels = tf.argmax(self.y, axis=1)

			# Predicted labels
			pred_labels = tf.argmax(logits, axis=1)

			correct = tf.equal(gold_labels, pred_labels)
			correct = tf.cast(correct, tf.float32)
			self._accuracy = tf.reduce_mean(correct)

			# Method for calculating precision, recall and F1 score
			# from https://stackoverflow.com/questions/35365007/tensorflow-precision-recall-f1-score-and-confusion-matrix
			self._TP = tf.count_nonzero(pred_labels*gold_labels, dtype=tf.float32)
			self._TN = tf.count_nonzero((pred_labels-1)*(gold_labels-1), dtype=tf.float32)
			self._FP = tf.count_nonzero(pred_labels*(gold_labels-1), dtype=tf.float32)
			self._FN = tf.count_nonzero((pred_labels-1)*gold_labels, dtype=tf.float32)

	@property
	def accuracy(self):
		return self._accuracy

	@property
	def TP(self):
		return self._TP

	@property
	def TN(self):
		return self._TN

	@property
	def FP(self):
		return self._FP

	@property
	def FN(self):
		return self._FN

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
	def words(self):
		return self._words

	@property
	def word_lens(self):
		return self._word_lens