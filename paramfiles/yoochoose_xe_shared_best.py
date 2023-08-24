from collections import OrderedDict
gru4rec_params = OrderedDict([
('loss', 'cross-entropy'),
('constrained_embedding', True),
('embedding', 0),
('final_act', 'softmax'),
('layers', [480]),
('n_epochs', 10),
('batch_size', 48),
('dropout_p_embed', 0.0),
('dropout_p_hidden', 0.2),
('learning_rate', 0.07),
('momentum', 0.0),
('n_sample', 2048),
('sample_alpha', 0.2),
('bpreg', 0.0),
('logq', 1.0)
])
