from enum import IntEnum
import numpy as np
import re
from numberer import Numberer
import random


# when true, all negative labels are collapsed into a single (arbitrary) label
COLLAPSE_NEGATIVE_CLASSES = True


class TaskType(IntEnum):
	Subtask_A = 1,		# binary clasification			- Inoffensive/Offensive
	Subtask_B = 2,		# binary classification			- (Inoffensive|Untargeted)/Targeted
	Subtask_C = 3,		# multi-class classification	- (Inoffensive|Untargeted)/Target(Individual|Group|Other)
	All = 4,			# multi-class classification	- Same as above


class OffenceClasses(IntEnum):
	Inoffensive = 0,
	Offensive = 1,
	Untargeted = 2,
	Targeted = 3,
	TargetIndividual = 4,
	TargetGroup = 5,
	TargetOther = 6


class DatasetFile:
	def __init__(self, file_path=None):
		self.file_path = file_path
		self.entries = list()		# list(cells)

		if file_path == None:
			self.file_path = "partitioned data"
			return

		with open(file_path, encoding="utf-8") as f:
			for (idx, line) in enumerate(f.read().split('\n')):
				if idx == 0: continue

				entries = line.strip().split('\t')
				if len(entries) == 0: continue

				self.entries.append(entries)

	def path(self):
		return self.file_path

	def lines(self):
		return self.entries

	def shuffle(self):
		random.shuffle(self.entries)

	def partition(self, partition):
		first = DatasetFile()
		second = DatasetFile()

		if partition >= 100 or partition <= 0:
			raise AssertionError("invalid partition " + str(partition));

		instances_first = int(partition / 100 * len(self.entries))
		self.shuffle()

		for idx in range(0, instances_first):
			first.entries.append(self.entries[idx])

		for idx in range(instances_first, len(self.entries)):
			second.entries.append(self.entries[idx])

		return (first, second)


class Dataset:
	def __init__(self):
		self.data = list()	# tuple(list(word tokens), list(list(char tokens)), class), word/character tokens are Numberer indices
		self.vocab = set()	# set of all (word token) ids in this dataset

	def put(self, text, text_chars, label):
		for id in text:
			self.vocab.add(id)
		self.data.append((text, text_chars, label))

	def as_list(self):
		return self.data

	def get_size(self):
		return len(self.data)

	def get_vocab_size(self):
		return len(self.vocab)

# loads up pre-trained embeddings from disk
class PretrainedEmbeddings:
	def __init__(self, dim=100, len=100):
		self.data = list()
		self.dim = dim
		self.length = len
		self.np_arr = np.zeros([self.length, self.dim], dtype='float32')

	def load(self, path, numberer, maxsize = -1):
		# reset defaults
		self.dim = 0
		self.length = 0
		self.np_arr = None

		counter = 0
		with open(path, encoding="utf-8") as f:
			for line in f.read().split('\n'):
				if (len(line) < 2): continue
				elif (maxsize > 0 and counter > maxsize): break

				splits = line.split(" ")
				if (self.dim == 0):
					self.dim = len(splits) - 1
					# the first index of the embedding list is not used (as Numberer ids start from 1)
					self.data.append(np.zeros((self.dim), dtype='float32'))
				elif len(splits) != self.dim + 1:
					print("embedding entry " + str(counter) + " has invalid splits (" + str(len(splits)) + ")" \
						+ "\nline: " + line)
					continue

				word = splits[0]
				id = numberer.number(word.lower())
				assert id == len(self.data)

				# using an intermediate list as we don't know the vocab size/row count beforehand
				vals = np.asarray(splits[1:], dtype='float32')
				self.data.append(vals)
				counter += 1

		# convert to NumPy array and discard the intermediate list
		self.np_arr = np.array(self.data)
		self.data = None
		self.length = counter + 1

		return counter

	def get_dim(self):
		return self.dim

	def get_data(self):
		return self.np_arr

	def get_size(self):
		return self.length


# loads the input data and embeddings
class Preprocessor:
	def __init__(self, task_type):
		self.task_type = task_type
		self.embeddings = PretrainedEmbeddings()
		self.numberer_word = Numberer()
		self.numberer_char = Numberer()
		self.numberer_label = Numberer()
		self.vocab = set()				# corresponding to the entire corpus
		self.train_set = None
		self.val_set = None
		self.vocab_sizes = None			# tuple(total vocab, embeddings only, training only, validation only)

	def preprocess_tweet(self, text):
		text = text.lower()
		# strip hashs from hastags and placeholder tokens
		to_strip = re.compile("(#|url|@user|@)")
		stripped = re.sub(to_strip, "", text)
		# naive word tokenization
		tokens = list(filter(None, stripped.split(" ")))

		return tokens

	def get_offence_class(self, label_subtask_a, label_subtask_b, label_subtask_c):
		final_label = None

		if self.task_type == TaskType.Subtask_A:
			if label_subtask_a == "NOT":
				final_label = OffenceClasses.Inoffensive
			elif label_subtask_a == "OFF":
				final_label = OffenceClasses.Offensive
		elif self.task_type == TaskType.Subtask_B:
			if label_subtask_b == "TIN":
				final_label = OffenceClasses.Targeted
			elif not COLLAPSE_NEGATIVE_CLASSES:
				if label_subtask_a == "NOT":
					final_label = OffenceClasses.Inoffensive
				else:
					final_label = OffenceClasses.Untargeted
			else:
				final_label = OffenceClasses.Untargeted
		else:
			if label_subtask_c == "IND":
				final_label = OffenceClasses.TargetIndividual
			elif label_subtask_c == "GRP":
				final_label = OffenceClasses.TargetGroup
			elif label_subtask_c == "OTH":
				final_label = OffenceClasses.TargetOther
			elif not COLLAPSE_NEGATIVE_CLASSES:
				if label_subtask_a == "NOT":
					final_label = OffenceClasses.Inoffensive
				else:
					final_label = OffenceClasses.Untargeted
			else:
				final_label = OffenceClasses.Untargeted

		if final_label == None:
			raise AssertionError("unknown label for dataset instance " + str(counter + 1) + " in file " + dataset_file.path() + "\t" \
				+ "labels: " + label_subtask_a + ", " + label_subtask_b + ", " + label_subtask_c)

		return final_label

	def get_max_labels(self):
		return self.numberer_label.max_number()

	def generate_dataset(self, dataset_file, numberer_word, numberer_char, numberer_label, maxsize = -1):
		dataset = Dataset()
		counter = 0

		for entry in dataset_file.lines():
			if len(entry) != 5:
				print("invalid dataset instance " + str(counter + 1) + " in file " + dataset_file.path() \
					+ "\tentry: " + str(entry))
				continue

			if maxsize > 0 and counter > maxsize:
				break

			#<id> <tweet> <labels>...
			text = entry[1]
			label_a = entry[2]
			label_b = entry[3]
			label_c = entry[4]

			collapsed_label_id = numberer_label.number(self.get_offence_class(label_a, label_b, label_c))

			# assign a unqiue id to all words and characters
			tokens = self.preprocess_tweet(text)
			word_ids = [numberer_word.number(word) for word in tokens]
			char_ids = list()
			for word in tokens:
				word_chars = list()
				for char in word:
					word_chars.append(numberer_char.number(char))
				char_ids.append(word_chars)

			for id in word_ids:
				self.vocab.add(id)

			dataset.put(word_ids, char_ids, collapsed_label_id)
			counter += 1

		return dataset

	def load(self, path_embed, dataset_file_train, dataset_file_val, maxsize_train = -1, maxsize_val = -1):
		# we load the pretrained embeddings first to initialize our basic word vocabulary
		# this way, the ids assigned to the words implicitly act as indices into the (pretrained) embedding matrix
		# words that don't have a pretrained embedding collect at the end of the embedding matrix
		# this lets us use random initializers for unknown words in the model
		vocab_embeddings = 0
		vocab_train_only = 0		# words in the training set that don't have an embedding
		vocab_val_only = 0			# same as above but for the validation set

		self.embeddings.load(path_embed, self.numberer_word)
		vocab_embeddings = self.numberer_word.max_number()
		self.train_set = self.generate_dataset(dataset_file_train, self.numberer_word, self.numberer_char, self.numberer_label, maxsize_train)
		vocab_after_train = self.numberer_word.max_number()
		vocab_train_only = vocab_after_train - vocab_embeddings
		self.val_set = self.generate_dataset(dataset_file_val, self.numberer_word, self.numberer_char, self.numberer_label, maxsize_val)
		vocab_after_val = self.numberer_word.max_number()
		vocab_val_only = vocab_after_val - vocab_after_train

		self.vocab_sizes = (len(self.vocab), vocab_embeddings, vocab_train_only, vocab_val_only)

	def get_embeddings(self):
		return self.embeddings

	def get_training_set(self):
		return self.train_set

	def get_validation_set(self):
		return self.val_set

	def get_vocab_size_trainonly(self):
		return self.vocab_sizes[2]

	def get_vocab_size_valonly(self):
		return self.vocab_sizes[3]

	def get_charvocab_size(self):
		return self.numberer_char.max_number()
