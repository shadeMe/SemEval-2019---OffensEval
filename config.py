class DefaultConfig:
	n_epochs = 20
	batch_size = 512
	l2_beta = 0.001

	use_char_embeddings = False
	char_embedding_size = 100
	char_rnn_max_timesteps = 240
	char_rnn_sizes = [140]
	char_rnn_output_dropout = 0.6
	char_rnn_state_dropout = 0.7

	word_rnn_max_timesteps = 50
	word_rnn_sizes = [25]
	word_rnn_output_dropout = 0.95
	word_rnn_state_dropout = 0.75