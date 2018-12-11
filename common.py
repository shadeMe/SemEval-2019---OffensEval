from enum import IntEnum
import numpy as np
import re
from numberer import Numberer
import random
import hashedindex as hi
from hashedindex import textparser
from nltk.corpus import stopwords


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
	def __init__(self, file_path=None, delimiter='\t'):
		self.file_path = file_path
		self.delimiter = delimiter
		self.entries = list()		# list(cells)

		if file_path == None:
			self.file_path = "partitioned data"
			return

		with open(file_path, encoding="utf-8") as f:
			for (idx, line) in enumerate(f.read().split('\n')):
				if idx == 0: continue

				entries = line.strip().split(self.delimiter)
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

	def merge(self, rhs):
		self.entries += rhs.entries;
		return self


class Dataset:
	def __init__(self):
		self.data = list()	# tuple(doc, list(word tokens), list(list(char tokens)), class), doc, word/character tokens are Numberer indices
		self.vocab = set()	# set of all (word token) ids in this dataset

	def put(self, doc, text, text_chars, label):
		for id in text:
			self.vocab.add(id)
		self.data.append((doc, text, text_chars, label))

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


# calculates tf-idf vectors for documents
class TfIdfVectorizer:
	def __init__(self, dim=140):
		self.dim = dim
		self.inverted_index = hi.HashedIndex()
		self.np_arr = None
		self.length = 0

	def add_document(self, doc_id, token_ids):
		for token in token_ids:
			self.inverted_index.add_term_occurrence(token, doc_id);

	def vectorize(self, numberer_doc, doc2token_map):
		assert self.np_arr == None
		doc_vecs = [np.zeros(self.dim, dtype="float32")]

		for i in range(1, numberer_doc.max_number()):	# index 0 is not used by the Numberer class
			token_ids = doc2token_map[i]
			if len(token_ids) > self.dim:
				raise AssertionError("document " + str(i) + " has " + str(len(token_ids)) + " tokens, max allowed = " + str(self.dim))

			tfidf_vec = np.zeros(self.dim, dtype="float32")
			for (idx, token) in enumerate(token_ids):
				tfidf_vec[idx] = self.inverted_index.get_tfidf(token, i, normalized=True)

			doc_vecs.append(tfidf_vec)

		assert len(doc_vecs) == numberer_doc.max_number()

		self.np_arr = np.array(doc_vecs)
		self.length = len(doc_vecs)

	def get_dim(self):
		return self.dim

	def get_data(self):
		return self.np_arr

	def get_size(self):
		return self.length


# loads the input data and embeddings
class Preprocessor:
	def __init__(self, task_type, config):
		self.task_type = task_type
		self.config = config
		self.embeddings = PretrainedEmbeddings()
		self.numberer_word = Numberer()
		self.numberer_char = Numberer()
		self.numberer_label = Numberer()
		self.numberer_doc = Numberer()
		self.tfidf = TfIdfVectorizer(config.tweet_max_words)
		self.vocab = set()						# corresponding to the entire corpus
		self.docs = dict()						# doc_id -> list(token_ids)
		self.train_set = None
		self.val_set = None
		self.vocab_sizes = None					# tuple(total vocab, embeddings only, training only, validation only)

	def preprocess_tweet(self, text):
		text = text.lower()
		# strip hashs from hastags and placeholder tokens
		to_strip = re.compile("(#|url|@user|@)")
		stripped = re.sub(to_strip, "", text)
		tokens = list(map(lambda x: x[0], textparser.word_tokenize(stripped,
															   stopwords.words('english') if self.config.remove_stopwords else [])))

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
			elif not self.config.collapse_negative_classes:
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
			elif label_subtask_c == "OTH" or label_subtask_c == "ORG":
				final_label = OffenceClasses.TargetOther
			elif not self.config.collapse_negative_classes:
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

	def generate_dataset(self, dataset_file, numberer_word, numberer_char, numberer_label, maxsize = -1):
		dataset = Dataset()
		counter = 0

		for entry in dataset_file.lines():
			if maxsize > 0 and counter > maxsize:
				break

			if len(entry) == 5:
				# OffensEval Training Dataset - <id> <tweet> <labels>...
				text = entry[1]
				label_a = entry[2]
				label_b = entry[3]
				label_c = entry[4]
			elif len(entry) == 4:
				# OffensEval Trial Dataset - <tweet> <labels>...
				text = entry[0]
				label_a = entry[1]
				label_b = entry[2]
				label_c = entry[3]
			elif len(entry) == 3:
				# TRAC Training Dataset - <id> <tweet> <label>
				# only for binary classification, must be preprocessed to use the OffsenEval tags
				text = entry[1]
				label_a = entry[2]
				label_b = ""
				label_c = ""
			else:
				print("invalid dataset instance " + str(counter + 1) + " in file " + dataset_file.path() \
					+ "\tentry: " + str(entry))
				continue

			collapsed_label_id = numberer_label.number(self.get_offence_class(label_a, label_b, label_c))
			document_id = self.numberer_doc.number(text);

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

			dataset.put(document_id, word_ids, char_ids, collapsed_label_id)
			self.tfidf.add_document(document_id, word_ids)
			self.docs[document_id] = word_ids

			counter += 1

		return dataset

	def load(self, path_embed, dataset_file_train, dataset_file_val, maxsize_train = -1, maxsize_val = -1):
		# we load the pretrained embeddings first to initialize our basic word vocabulary
		# this way, the ids assigned to the words implicitly act as indices into the (pretrained) embedding matrix
		# words that don't have a pretrained embedding collect at the end of the embedding matrix
		# this lets us use specific initializers for unknown words in the model
		vocab_embeddings = 0
		vocab_train_only = 0		# words in the training set that don't have an embedding
		vocab_val_only = 0			# same as above but for the validation set

		print("Loading embeddings...")
		self.embeddings.load(path_embed, self.numberer_word)
		vocab_embeddings = self.numberer_word.max_number()

		print("Loading training data...")
		self.train_set = self.generate_dataset(dataset_file_train, self.numberer_word, self.numberer_char, self.numberer_label, maxsize_train)
		vocab_after_train = self.numberer_word.max_number()
		vocab_train_only = vocab_after_train - vocab_embeddings

		print("Loading validation data...")
		self.val_set = self.generate_dataset(dataset_file_val, self.numberer_word, self.numberer_char, self.numberer_label, maxsize_val)
		vocab_after_val = self.numberer_word.max_number()
		vocab_val_only = vocab_after_val - vocab_after_train

		self.vocab_sizes = (len(self.vocab), vocab_embeddings, vocab_train_only, vocab_val_only)

		print("Calculating TF-IDF matrix...")
		self.tfidf.vectorize(self.numberer_doc, self.docs)


	def get_tfidf(self):
		return self.tfidf

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

	def get_max_labels(self):
		return self.numberer_label.max_number()

	def get_max_docs(self):
		return self.numberer_doc.max_number()