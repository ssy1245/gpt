out_dir = 'out-rocstories'

eval_interval = 200
eval_iters = 200
log_interval = 10

dataset = 'rocstories'

batch_size = 32
gradient_accumulation_steps = 2
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

# Optimization
learning_rate = 2e-4
max_iters = 25000
lr_decay_iters = 25000
min_lr = 2e-5

beta2 = 0.95
warmup_iters = 1000
beta2 = 0.95
warmup_iters = 1000