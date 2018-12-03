class DefaultConfig:
	n_epochs = 20
	batch_size = 512
	l2_beta = 0.001

	char_embedding_size = 50
	char_rnn_max_timesteps = 20
	char_rnn_sizes = [10]
	char_rnn_output_dropout = 0.9
	char_rnn_state_dropout = 0.9

	word_rnn_max_timesteps = 140
	word_rnn_sizes = [100]
	word_rnn_output_dropout = 0.9
	word_rnn_state_dropout = 0.8