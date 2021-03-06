from enum import IntEnum
import numpy as np
import re
from numberer import Numberer
import random
import hashedindex as hi
from hashedindex import textparser
from nltk.corpus import stopwords
from collections import Counter

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
	def __init__(self, file_path=None, delimiter='\t', encoding='utf-8'):
		self.file_path = file_path
		self.delimiter = delimiter
		self.entries = list()		# list(cells)

		if file_path == None:
			self.file_path = "partitioned data"
			return

		with open(file_path, encoding=encoding) as f:
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

	def keep_first(self, num_entries_to_keep):
		assert num_entries_to_keep > 0

		if num_entries_to_keep >= len(self.entries):
			return

		self.entries = self.entries[0:num_entries_to_keep]


class Dataset:
	def __init__(self):
		self.data = list()	# tuple(doc, list(word tokens), list(list(char tokens)), class), doc, word/character tokens are Numberer indices
		self.vocab = set()	# set of all (word token) ids in this dataset
		self.label_counts = Counter()

	def put(self, doc, text, text_chars, label):
		for id in text:
			self.vocab.add(id)
		self.data.append((doc, text, text_chars, label))
		self.label_counts[label] += 1

	def as_list(self):
		return self.data

	def get_size(self):
		return len(self.data)

	def get_vocab_size(self):
		return len(self.vocab)

	def get_class_ratios(self, numberer_labels, complement=False):
		ratios = np.zeros(numberer_labels.max_number(), dtype="float32")
		for i in range(0, numberer_labels.max_number()):
			ratios[i] = self.label_counts[i] / self.get_size() if not complement else 1 - self.label_counts[i] / self.get_size()
		return ratios

	def print_distribution(self, numberer_labels):
		for i in range(0, numberer_labels.max_number()):
			samples = self.label_counts[i]
			label = numberer_labels.value(i)
			print("\t" + str(label) + ": " + str(samples) + " samples (" + str(samples / self.get_size() * 100.) + "%)")



# loads up pre-trained embeddings from disk
class PretrainedEmbeddings:
	def __init__(self, dim=100, len=1):
		self.data = list()
		self.dim = dim
		self.length = len
		self.np_arr = np.zeros([self.length, self.dim], dtype='float32')

	def read_lines(self, file):
		while True:
			line = file.readline()
			if line != "":
				yield line
			else:
				break

	def load(self, path, numberer, maxsize = -1):
		# reset defaults
		self.dim = 0
		self.length = 0
		self.np_arr = None

		counter = 0
		with open(path, encoding="utf-8") as f:
			lines = self.read_lines(f)
			for line in lines:
				if (len(line) < 2): continue
				elif (maxsize > 0 and counter >= maxsize): break

				splits = line.split(" ")
				if len(splits) < 3:
					continue;

				if (self.dim == 0):
					self.dim = len(splits) - 1
				elif len(splits) != self.dim + 1:
					print("invalid splits (found " + str(len(splits)) + ", expected " + str(self.dim + 1) + ")\nline: " + line)
					continue

				word = splits[0]
				id = numberer.number(word.lower())
				if id == len(self.data):
					print("word '" + word.lower() + "' has multiple embeddings (case-sensitive?)")
					continue

				# using an intermediate list as we don't know the vocab size/row count beforehand
				vals = np.asarray(splits[1:], dtype='float32')
				self.data.append(vals)
				counter += 1

		# convert to NumPy array and discard the intermediate list
		self.np_arr = np.array(self.data)
		self.data = None
		self.length = counter

		return counter

	def get_dim(self):
		return self.dim

	def get_data(self):
		return self.np_arr

	def get_size(self):
		return self.length


# calculates tf-idf vectors for documents
class TfIdfVectorizer:
	def __init__(self, dim):
		self.dim = dim
		self.inverted_index = hi.HashedIndex()
		self.np_arr = None
		self.length = 0

	def add_document(self, doc_id, token_ids):
		for token in token_ids:
			self.inverted_index.add_term_occurrence(token, doc_id);

	def vectorize(self, numberer_doc, doc2token_map):
		assert self.np_arr == None
		doc_vecs = []

		for i in range(0, numberer_doc.max_number()):
			token_ids = doc2token_map[i]
			if len(token_ids) > self.dim:
				# truncate the rest
				token_ids = token_ids[0:self.dim - 1]

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


# calculates sentiment vectors for documents
class VaderSentimentVectorizer:
	def __init__(self, dim):
		self.dim = dim
		self.np_arr = None
		self.length = 0
		self.lexicon = dict()


	def load_lexicon(self, dataset_lexicon):
		for entry in dataset_lexicon.lines():
			self.lexicon[entry[0]] = float(entry[2])

	def vectorize(self, numberer_doc, numberer_word, doc2token_map):
		assert self.np_arr == None
		doc_vecs = []

		for i in range(0, numberer_doc.max_number()):
			token_ids = doc2token_map[i]
			if len(token_ids) > self.dim:
				token_ids = token_ids[0:self.dim - 1]

			sentiment_vec = np.zeros(self.dim, dtype="float32")
			for (idx, token) in enumerate(token_ids):
				polarity = self.lexicon.get(numberer_word.value(token))
				if polarity != None:
					sentiment_vec[idx] = polarity

			doc_vecs.append(sentiment_vec)

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
		self.tfidf = TfIdfVectorizer(config.doc_vector_size)
		self.sentiment = VaderSentimentVectorizer(config.doc_vector_size)
		self.vocab = set()						# corresponding to the entire corpus
		self.docs = dict()						# doc_id -> list(token_ids)
		self.train_set = None
		self.val_set = None
		self.vocab_sizes = None					# tuple(total vocab, embeddings only, training only, validation only)

	def preprocess_tweet(self, text):
		text = text.lower()
		if not self.config.remove_hash_tags_and_mentions:
			stripped = re.sub(r'\burl\b', '', text)
		else:
			stripped = re.sub(r'\burl\b', '', text)
			stripped = re.sub(r'(\b|\s)([@#][\w_-]+)', '', stripped)

		tokens = list(map(lambda x: x[0], textparser.word_tokenize(stripped,
															   stopwords.words('english') if self.config.remove_stopwords else [])))

		return tokens

	def get_offence_class(self, dataset_file, text, label_subtask_a, label_subtask_b, label_subtask_c):
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

		#if final_label == None:
		#	print("unknown label for dataset instance '" + text + "' in file " + dataset_file.path() + "\t" + "labels: " + label_subtask_a + ", " + label_subtask_b + ", " + label_subtask_c)

		return final_label

	def generate_char_ids(self, tokens, numberer_char):
		char_ids = list()
		for word in tokens:
			word_chars = list()
			for char in word:
				word_chars.append(numberer_char.number(char))
			char_ids.append(word_chars)

		return char_ids

	def generate_char_ngram_ids(self, tokens, numberer_char):
		if self.config.char_ngram_size < 1:
			raise AssertionError("character n-gram size must be greater than zero")

		n = self.config.char_ngram_size
		# both @ and # should be stripped from the token input in the preprocessing step
		pad_begin = ("@", ) * (n - 1)
		pad_end = "#"

		sent_char_ngram_ids = []
		for token in tokens:
			padded_seq = list(pad_begin)
			padded_seq.extend([char for char in token])
			padded_seq.append(pad_end)
			char_ngrams = zip(*[padded_seq[i:] for i in range(1 + len(pad_begin))])

			char_ngram_ids = [numberer_char.number(char_ngram) for char_ngram in char_ngrams]
			sent_char_ngram_ids.append(char_ngram_ids)

		return sent_char_ngram_ids


	def generate_dataset(self, dataset_file, numberer_word, numberer_char, numberer_label, maxsize = -1):
		dataset = Dataset()
		counter = 0

		for entry in dataset_file.lines():
			if maxsize > 0 and counter > maxsize:
				break

			if len(entry) == 5:
				# OffensEval/HatEval Training Dataset - <id> <tweet> <labels>...
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
				# multi-line entries are skipped
				text = entry[1]
				label_a = entry[2]
				label_b = ""
				label_c = ""
			else:
		#		print("invalid dataset instance " + str(counter + 1) + " in file " + dataset_file.path() \
		#			+ "\tentry: " + str(entry))
				continue

			collapsed_label_id = numberer_label.number(self.get_offence_class(dataset_file, text, label_a, label_b, label_c))
			if collapsed_label_id == None:
				continue;

			document_id = self.numberer_doc.number(text);

			# assign a unqiue id to all words and characters
			tokens = self.preprocess_tweet(text)
			word_ids = [numberer_word.number(word) for word in tokens]

			if self.config.use_char_ngrams:
				char_ids = self.generate_char_ngram_ids(tokens, numberer_char)
			else:
				char_ids = self.generate_char_ids(tokens, numberer_char)

			for id in word_ids:
				self.vocab.add(id)

			dataset.put(document_id, word_ids, char_ids, collapsed_label_id)
			self.tfidf.add_document(document_id, word_ids)
			self.docs[document_id] = word_ids

			counter += 1

		return dataset

	def load(self, path_embed, path_sent_lexicon, dataset_file_train, dataset_file_val, maxsize_train = -1, maxsize_val = -1):
		# we load the pretrained embeddings first to initialize our basic word vocabulary
		# this way, the ids assigned to the words implicitly act as indices into the (pretrained) embedding matrix
		# words that don't have a pretrained embedding collect at the end of the embedding matrix
		# this lets us use specific initializers for unknown words in the model
		vocab_embeddings = 0
		vocab_train_only = 0		# words in the training set that don't have an embedding
		vocab_val_only = 0			# same as above but for the validation set

		print("Loading embeddings...")
		self.embeddings.load(path_embed, self.numberer_word)
		print("\tParsed " + str(self.embeddings.get_size()) + " words of dimension " + str(self.embeddings.get_dim()));
		vocab_embeddings = self.numberer_word.max_number()

		print("Loading sentiment lexicon...")
		self.sentiment.load_lexicon(DatasetFile(path_sent_lexicon))

		print("Loading training data...")
		self.train_set = self.generate_dataset(dataset_file_train, self.numberer_word, self.numberer_char, self.numberer_label, maxsize_train)
		vocab_after_train = self.numberer_word.max_number()
		vocab_train_only = vocab_after_train - vocab_embeddings
		print("\tParsed " + str(self.train_set.get_size()) + " instances")
		self.train_set.print_distribution(self.numberer_label)

		print("Loading validation data...")
		self.val_set = self.generate_dataset(dataset_file_val, self.numberer_word, self.numberer_char, self.numberer_label, maxsize_val)
		vocab_after_val = self.numberer_word.max_number()
		vocab_val_only = vocab_after_val - vocab_after_train
		print("\tParsed " + str(self.val_set.get_size()) + " instances")
		self.val_set.print_distribution(self.numberer_label)

		self.vocab_sizes = (len(self.vocab), vocab_embeddings, vocab_train_only, vocab_val_only)

		print("Calculating TF-IDF matrix...")
		self.tfidf.vectorize(self.numberer_doc, self.docs)

		print("Calculating sentiment matrix...")
		self.sentiment.vectorize(self.numberer_doc, self.numberer_word, self.docs)


	def get_tfidf(self):
		return self.tfidf

	def get_sentiment(self):
		return self.sentiment

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

	def get_task_type(self):
		return self.task_type