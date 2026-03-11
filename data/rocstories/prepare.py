import os
import tiktoken
import numpy as np
import os
from datasets import load_dataset

from datasets import load_dataset
import tiktoken
import numpy as np

dataset = load_dataset("mintujupally/ROCStories")

stories = dataset["train"]["text"]

# story-level split
n = len(stories)
train_stories = stories[:int(n*0.9)]
val_stories = stories[int(n*0.9):]

# concatenate
train_text = "\n\n".join(train_stories)
val_text = "\n\n".join(val_stories)

# tokenizer
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_text)
val_ids = enc.encode_ordinary(val_text)

print("train tokens:", len(train_ids))
print("val tokens:", len(val_ids))

# save
np.array(train_ids, dtype=np.uint16).tofile("train.bin")
np.array(val_ids, dtype=np.uint16).tofile("val.bin")