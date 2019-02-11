class DefaultConfig:
	def __init__(self):
		self.remove_hash_tags_and_mentions = True
		self.remove_stopwords = True
		self.collapse_negative_classes = True		# when true, all negative labels are collapsed into a single (arbitrary) label

		self.n_epochs = 50
		self.batch_size = 512
		self.doc_vector_size = 30					# corresponds to the (max) number of words allowed in the tweet
		self.use_weighted_loss = True
		self.use_l2_regularization = True
		self.l2_beta = 0.001

		self.word_rnn_max_timesteps = 100
		self.word_rnn_sizes = [100, 100]
		self.word_rnn_output_dropout = 0.9
		self.word_rnn_state_dropout = 0.85

		self.use_final_hidden_layer = False
		self.final_hidden_layer_size = 100
		self.final_hidden_layer_dropout = 0.95

		self.use_tfidf_vectors = False
		self.use_sentiment_vectors = False

		self.use_char_embeddings = True
		self.use_char_ngrams = True
		self.char_ngram_size = 3
		self.char_embedding_size = 100
		self.char_rnn_max_timesteps = 100
		self.char_rnn_sizes = [100, 100]
		self.char_rnn_output_dropout = 0.9
		self.char_rnn_state_dropout = 0.85

		self.early_stopping = True


	def print(self):
		print("Config:\t" + str(vars(self)))
		print("\n\n")