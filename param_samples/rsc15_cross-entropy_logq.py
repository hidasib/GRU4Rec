from collections import OrderedDict
gru4rec_params = OrderedDict([
('layers', [100]),
('loss', 'cross-entropy'),
('final_act', 'softmax'),
('hidden_act', 'tanh'),
('adapt', 'adagrad'),
('n_epochs', 10),
('batch_size', 32),
('dropout_p_embed', 0.0),
('dropout_p_hidden', 0.4),
('learning_rate', 0.2),
('momentum', 0.2),
('sample_alpha', 0.5),
('n_sample', 2048),
('logq', 1.0),
('constrained_embedding', True)
])