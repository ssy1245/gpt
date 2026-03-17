from datasets import load_dataset
import tiktoken
import numpy as np


from datasets import load_dataset
import tiktoken
import numpy as np
import random

# =========================
# 1. Load dataset
# =========================
dataset = load_dataset("mintujupally/ROCStories")

train_stories = list(dataset["train"]["text"])
test_stories = list(dataset["test"]["text"])

print(f"Total train stories: {len(train_stories)}")
print(f"Total test stories: {len(test_stories)}")

print("Example story:")
print(train_stories[0])
print("-" * 50)

# =========================
# 2. Shuffle (VERY IMPORTANT)
# =========================
random.seed(42)
random.shuffle(train_stories)

# =========================
# 3. Tokenizer
# =========================
enc = tiktoken.get_encoding("gpt2")
eot_id = enc.eot_token

# =========================
# 4. Encode train
# =========================
train_ids = []

for story in train_stories:
    # add structure signal
    text = "Story: " + story.strip()

    ids = enc.encode_ordinary(text)
    train_ids.extend(ids)
    train_ids.append(eot_id)

train_ids = np.array(train_ids, dtype=np.uint16)
print(f"Train tokens: {len(train_ids):,}")
train_ids.tofile("train.bin")

# =========================
# 5. Encode test
# =========================
test_ids = []

for story in test_stories:
    text = "Story: " + story.strip()

    ids = enc.encode_ordinary(text)
    test_ids.extend(ids)
    test_ids.append(eot_id)

test_ids = np.array(test_ids, dtype=np.uint16)
print(f"Test tokens: {len(test_ids):,}")
test_ids.tofile("test.bin")

# =========================
# 6. Sanity check
# =========================
print("-" * 50)
print("Sanity check:")

sample = "Story: " + train_stories[0]
ids = enc.encode_ordinary(sample) + [eot_id]
decoded = enc.decode(ids)

print(decoded)
print("Last token:", ids[-1], "| Should be eot:", eot_id)
print("Check passed:", ids[-1] == eot_id)