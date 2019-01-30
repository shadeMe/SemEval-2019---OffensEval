class DefaultConfig:
	# all dropout values are keep probabilities
	def __init__(self):
		self.remove_hash_tags_and_mentions = False
		self.remove_stopwords = False
		self.collapse_negative_classes = True		# when true, all negative labels are collapsed into a single (arbitrary) label

		self.n_epochs = 25
		self.batch_size = 512
		self.l2_beta = 0.001
		self.doc_vector_size = 35					# corresponds to the (max) number of words allowed in the tweet

		self.word_rnn_max_timesteps = 100
		self.word_rnn_sizes = [50]
		self.word_rnn_output_dropout = 0.75
		self.word_rnn_state_dropout = 0.5

		self.use_final_hidden_layer = False
		self.final_hidden_layer_size = 100
		self.final_hidden_layer_dropout = 0.75

		self.use_tfidf_vectors = False
		self.use_sentiment_vectors = False

		self.use_char_embeddings = False
		self.use_char_ngrams = True
		self.char_ngram_size = 3
		self.char_embedding_size = 50
		self.char_rnn_max_timesteps = 50
		self.char_rnn_sizes = [25, 10]
		self.char_rnn_output_dropout = 0.75
		self.char_rnn_state_dropout = 0.5

		self.stop_on_overfitting = False


	def print(self):
		print("Config:\t" + str(vars(self)))
		print("\n\n")