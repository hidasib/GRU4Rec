from collections import OrderedDict
gru4rec_params = OrderedDict([
('loss', 'bpr-max'),
('constrained_embedding', True),
('embedding', 0),
('final_act', 'elu-1'),
('layers', [512]),
('n_epochs', 10),
('batch_size', 144),
('dropout_p_embed', 0.35),
('dropout_p_hidden', 0.0),
('learning_rate', 0.05),
('momentum', 0.4),
('n_sample', 2048),
('sample_alpha', 0.2),
('bpreg', 1.85),
('logq', 0.0)
])
