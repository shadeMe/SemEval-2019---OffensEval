from enum import Enum
import os
import sys

import numpy as np
import tensorflow as tf

from config import DefaultConfig
from model import Model, Phase
from common import *


def generate_instances(
		data,
		max_label,
		max_timesteps,
		max_char_timesteps,
		batch_size=128):
	n_batches = data.get_size() // batch_size
	data = data.as_list()

	# We are discarding the last batch for now, for simplicity.
	labels = np.zeros(
		shape=(
			n_batches,
			batch_size,
			max_label),
		dtype=np.float32)
	# sentence lengths
	lengths = np.zeros(
		shape=(
			n_batches,
			batch_size),
		dtype=np.int32)
	# sentence representations with word IDs
	sents = np.zeros(
		shape=(
			n_batches,
			batch_size,
			max_timesteps),
		dtype=np.int32)
	# word lengths in number of chars
	word_lengths = np.zeros(
		shape=(
			n_batches,
			batch_size,
			max_timesteps),
		dtype=np.int32)
	# word representation with IDs of chars
	words = np.zeros(
		shape=(
			n_batches,
			batch_size,
			max_timesteps,
			max_char_timesteps),
		dtype=np.int32)

	for batch in range(n_batches):
		for idx in range(batch_size):
			(tokens, token_chars, sentiment) = data[(batch * batch_size) + idx]
			label = sentiment
			labels[batch, idx, label] = 1

			# Sequence
			timesteps = min(max_timesteps, len(tokens))

			# Sequence length (time steps)
			lengths[batch, idx] = timesteps

			# Sentences
			sents[batch, idx, :timesteps] = tokens[:timesteps]

			# Words
			for word_id in range(timesteps):
				char_timesteps = min(max_char_timesteps, len(token_chars[word_id]))
				word_lengths[batch, idx, word_id] = char_timesteps
				words[batch, idx, word_id, :char_timesteps] = token_chars[word_id][:char_timesteps]

	return (sents, lengths, labels, word_lengths, words)


def train_model(preproc, config, train_batches, validation_batches):
	train_batches, train_lens, train_labels, train_word_lengths, train_words = train_batches
	validation_batches, validation_lens, validation_labels, validation_word_lengths, validation_words = validation_batches

	with tf.Session() as sess:
		with tf.variable_scope("model", reuse=False):
			train_model = Model(
				preproc,
				config,
				train_batches,
				train_lens,
				train_labels,
				train_word_lengths,
				train_words,
				phase=Phase.Train)

		with tf.variable_scope("model", reuse=True):
			validation_model = Model(
				preproc,
				config,
				validation_batches,
				validation_lens,
				validation_labels,
				validation_word_lengths,
				validation_words,
				phase=Phase.Validation)

		sess.run(tf.global_variables_initializer())

		prev_validation_loss = 0.0
		print("===============================================================================================================")
		print("Epoch\tTrain Loss\tVal Loss\tDelta\t\tAccuracy\tPrecision\tRecall\t\tF1");
		print("===============================================================================================================")

		for epoch in range(config.n_epochs):
			train_loss = 0.0
			validation_loss = 0.0
			accuracy = 0.0
			TP = 0.0
			TN = 0.0
			FP = 0.0
			FN = 0.0

			# train on all batches.
			for batch in range(train_batches.shape[0]):
				loss, _ = sess.run([train_model.loss, train_model.train_op], {
					train_model.x: train_batches[batch], train_model.lens: train_lens[batch], train_model.word_lens: train_word_lengths[batch], train_model.words: train_words[batch], train_model.y: train_labels[batch]})
				train_loss += loss

			# validation on all batches.
			for batch in range(validation_batches.shape[0]):
				loss, acc, batch_tp, batch_tn, batch_fp, batch_fn = sess.run([validation_model.loss, validation_model.accuracy, validation_model.TP, validation_model.TN, validation_model.FP, validation_model.FN], {
					validation_model.x: validation_batches[batch], validation_model.lens: validation_lens[batch], validation_model.word_lens: validation_word_lengths[batch], validation_model.words: validation_words[batch], validation_model.y: validation_labels[batch]})
				validation_loss += loss
				accuracy += acc
				TP += batch_tp
				TN += batch_tn
				FP += batch_fp
				FN += batch_fn

			train_loss /= train_batches.shape[0]
			validation_loss /= validation_batches.shape[0]
			accuracy /= validation_batches.shape[0]
			precision = TP / (TP + FP)
			recall = TP / (TP + FN)
			f1 = 2 * precision * recall / (precision + recall)


			print("%d\t%.2f\t\t%.2f\t\t%.4f\t\t%.2f\t\t%.4f\t\t%.4f\t\t%.4f" %
				(epoch, train_loss, validation_loss, prev_validation_loss - validation_loss, accuracy * 100, precision, recall, f1))

			prev_validation_loss = validation_loss

# Usage: python train.py TASK WORD_EMBEDDINGS TRAIN_DATA TEST_DATA
#	where TASK = <A, B, C>

# Pre-trained word embeddings used:
#	English: https://nlp.stanford.edu/projects/glove/

DEFAULT_TRAINING_DATA_PARTITION = 80
DEFAULT_TASK_TYPE = TaskType.Subtask_C

def print_usage():
	print("Usage: python train.py WORD_EMBEDDINGS TRAIN_DATA TEST_DATA\n\twhere TASK = <A, B, C>\n\n")

if __name__ == "__main__":
	print("\n\n\n\n\n\n")

	if len(sys.argv) != 5:
		print_usage()

		path_embed = "C:\\Users\\shadeMe\\Documents\\ML\\Embeddings\\glove.twitter.27B.100d.txt"

		(train, test) = DatasetFile("Data\\offenseval-training-v1.tsv").partition(DEFAULT_TRAINING_DATA_PARTITION)
		task_type = DEFAULT_TASK_TYPE
	else:
		task_type = sys.argv[1]
		if task_type == "A" or task_type == "a":
			task_type = TaskType.Subtask_A
		elif task_type == "B" or task_type == "b":
			task_type = TaskType.Subtask_B
		elif task_type == "C" or task_type == "c":
			task_type = TaskType.Subtask_C
		else:
			print_usage()
			sys.exit()

		path_embed = sys.argv[2]
		path_train = sys.argv[3]
		path_val = sys.argv[4]

		train = DatasetFile(path_train)
		test = DatasetFile(path_test)

	config = DefaultConfig()
	preproc = Preprocessor(task_type)

	preproc.load(path_embed, train, test)
	data_train = preproc.get_training_set()
	data_validation = preproc.get_validation_set()

	# Generate batches
	train_batches = generate_instances(
		data_train,
		preproc.get_max_labels(),
		config.max_timesteps,
		config.max_char_timesteps,
		batch_size=config.batch_size)
	validation_batches = generate_instances(
		data_validation,
		preproc.get_max_labels(),
		config.max_timesteps,
		config.max_char_timesteps,
		batch_size=config.batch_size)

	# Train the model
	train_model(preproc, config, train_batches, validation_batches)