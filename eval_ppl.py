import math
import torch
import numpy as np
from contextlib import nullcontext
from model import GPT, GPTConfig

# =====================
# 配置（你只需要改这里）
# =====================
out_dir = 'out-rocstories'
test_bin = 'data/rocstories/test.bin'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# =====================
# 加载模型
# =====================
ckpt_path = f"{out_dir}/ckpt.pt"
checkpoint = torch.load(ckpt_path, map_location=device)

gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

# AMP
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# =====================
# 读取 test.bin
# =====================
test_data = np.memmap(test_bin, dtype=np.uint16, mode='r')

block_size = model.config.block_size
stride = block_size  # 可以改成 block_size//2 更精确

# =====================
# 计算 PPL
# =====================
total_nll = 0.0
total_tokens = 0

with torch.no_grad():
    with ctx:
        for i in range(0, len(test_data) - block_size, stride):
            x = torch.from_numpy(test_data[i:i+block_size].astype(np.int64)).to(device)
            y = torch.from_numpy(test_data[i+1:i+1+block_size].astype(np.int64)).to(device)

            x = x.unsqueeze(0)  # (1, T)
            y = y.unsqueeze(0)

            _, loss = model(x, y)

            n_tokens = y.numel()
            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens

avg_loss = total_nll / total_tokens
ppl = math.exp(avg_loss)

print("===== Test PPL =====")
print(f"tokens     : {total_tokens}")
print(f"avg_loss   : {avg_loss:.4f}")
print(f"ppl        : {ppl:.2f}")