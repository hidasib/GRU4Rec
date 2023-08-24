from collections import OrderedDict
gru4rec_params = OrderedDict([
('loss', 'bpr-max'),
('constrained_embedding', True),
('embedding', 0),
('final_act', 'elu-0.5'),
('layers', [224]),
('n_epochs', 10),
('batch_size', 80),
('dropout_p_embed', 0.5),
('dropout_p_hidden', 0.05),
('learning_rate', 0.05),
('momentum', 0.4),
('n_sample', 2048),
('sample_alpha', 0.4),
('bpreg', 1.95),
('logq', 0.0)
])
