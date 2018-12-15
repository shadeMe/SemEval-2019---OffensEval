class DefaultConfig:
	def __init__(self):
		self.remove_hash_tags_and_mentions = False
		self.remove_stopwords = True
		self.collapse_negative_classes = True		# when true, all negative labels are collapsed into a single (arbitrary) label

		self.n_epochs = 30
		self.batch_size = 512
		self.l2_beta = 0.001
		self.final_hidden_layer_size = 100
		self.final_hidden_layer_dropout = 0.9

		self.concat_uni_rnn_size = -1
		self.concat_uni_rnn_output_dropout = 0.8
		self.concat_uni_rnn_state_dropout = 0.8

		self.use_tfidf_vectors = False
		self.tf_idf_vector_size = 30

		self.use_char_embeddings = True
		self.char_embedding_size = 50
		self.char_rnn_max_timesteps = 70
		self.char_rnn_sizes = [50]
		self.char_rnn_output_dropout = 0.8
		self.char_rnn_state_dropout = 0.7

		self.use_word_embeddings = True
		self.word_rnn_max_timesteps = 100
		self.word_rnn_sizes = [50]
		self.word_rnn_output_dropout = 0.75
		self.word_rnn_state_dropout = 0.9


	def print(self):
		print("Config:\t" + str(vars(self)))
		print("\n\n")