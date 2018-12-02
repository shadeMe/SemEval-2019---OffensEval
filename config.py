class DefaultConfig:
	n_epochs = 20
	batch_size = 512
	input_dropout = 1.0
	hidden_dropout = 0.75
	char_embedding_size = 50
	max_timesteps = 100
	forward_cell_units = 50
	backward_cell_units = 50
	l2_beta = 0.001

	max_char_timesteps = 20