out_dir = 'out-rocstories'
eval_interval = 250
eval_iters = 100
log_interval = 10

dataset = 'rocstories'

gradient_accumulation_steps = 1
batch_size = 8
block_size = 256

n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.2

learning_rate = 5e-4
max_iters = 12000
lr_decay_iters = 12000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100