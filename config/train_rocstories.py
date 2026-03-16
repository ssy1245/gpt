out_dir = 'out-rocstories'
eval_interval = 250
eval_iters = 200
log_interval = 10

dataset = 'rocstories'

gradient_accumulation_steps = 1
batch_size = 32
block_size = 512

n_layer = 10
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 3e-4
max_iters = 20000
lr_decay_iters = 18000
min_lr = 3e-5
beta2 = 0.99
warmup_iters = 200