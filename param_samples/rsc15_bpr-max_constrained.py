from collections import OrderedDict
gru4rec_params = OrderedDict([
('layers', [100]),
('loss', 'bpr-max'),
('final_act', 'elu-0.5'),
('hidden_act', 'tanh'),
('adapt', 'adagrad'),
('n_epochs', 10),
('batch_size', 32),
('dropout_p_embed', 0.0),
('dropout_p_hidden', 0.0),
('learning_rate', 0.2),
('momentum', 0.1),
('sample_alpha', 0.0),
('n_sample', 2048),
('bpreg', 0.5),
('constrained_embedding', True)
])
