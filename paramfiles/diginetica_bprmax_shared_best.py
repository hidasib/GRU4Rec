from collections import OrderedDict
gru4rec_params = OrderedDict([
('loss', 'bpr-max'),
('constrained_embedding', True),
('embedding', 0),
('final_act', 'elu-1'),
('layers', [512]),
('n_epochs', 10),
('batch_size', 128),
('dropout_p_embed', 0.5),
('dropout_p_hidden', 0.3),
('learning_rate', 0.05),
('momentum', 0.15),
('n_sample', 2048),
('sample_alpha', 0.3),
('bpreg', 0.9),
('logq', 0.0)
])
