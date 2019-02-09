class DefaultConfig:
	def __init__(self):
		self.remove_hash_tags_and_mentions = False
		self.remove_stopwords = True
		self.collapse_negative_classes = True		# when true, all negative labels are collapsed into a single (arbitrary) label

		self.n_epochs = 200
		self.batch_size = 512
		self.doc_vector_size = 25					# corresponds to the (max) number of words allowed in the tweet
		self.use_weighted_loss = True
		self.use_l2_regularization = True
		self.l2_beta = 0.01

		self.word_rnn_max_timesteps = 50
		self.word_rnn_sizes = [100, 100]
		self.word_rnn_output_dropout = 0.75
		self.word_rnn_state_dropout = 0.5

		self.use_final_hidden_layer = False
		self.final_hidden_layer_size = 100
		self.final_hidden_layer_dropout = 0.75

		self.use_tfidf_vectors = False
		self.use_sentiment_vectors = False

		self.use_char_embeddings = True
		self.use_char_ngrams = True
		self.char_ngram_size = 3
		self.char_embedding_size = 50
		self.char_rnn_max_timesteps = 50
		self.char_rnn_sizes = [50, 50]
		self.char_rnn_output_dropout = 0.75
		self.char_rnn_state_dropout = 0.5


	def print(self):
		print("Config:\t" + str(vars(self)))
		print("\n\n")