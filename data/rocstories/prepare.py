from datasets import load_dataset
import tiktoken
import numpy as np


# 1. Load dataset
dataset = load_dataset("mintujupally/ROCStories")
stories = dataset["train"]["text"]

print(f"Total stories: {len(stories)}")
print("Example story:")
print(stories[0])
print("-" * 50)

# 2. Story-level split
n = len(stories)
split_idx = int(n * 0.9)
train_stories = stories[:split_idx]
val_stories = stories[split_idx:]

print(f"Train stories: {len(train_stories)}")
print(f"Val stories:   {len(val_stories)}")


# 3. Tokenizer
enc = tiktoken.get_encoding("gpt2")
eot_id = enc.eot_token  # GPT-2 true <|endoftext|> token id

print(f"GPT-2 EOT token id: {eot_id}")


# 4. Encode each story, then append true EOT token
train_ids = []
for story in train_stories:
    # story text -> ordinary tokens
    story_ids = enc.encode_ordinary(story)
    train_ids.extend(story_ids)
    # append true EOS / EOT token
    train_ids.append(eot_id)

val_ids = []
for story in val_stories:
    story_ids = enc.encode_ordinary(story)
    val_ids.extend(story_ids)
    val_ids.append(eot_id)

# 5. Convert to uint16 and save
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens:   {len(val_ids):,}")

train_ids.tofile("train.bin")
val_ids.tofile("val.bin")

print("Saved train.bin and val.bin")

# 6. Sanity check
print("-" * 50)
print("Sanity check on first training example:")

first_story_ids = enc.encode_ordinary(train_stories[0]) + [eot_id]
decoded = enc.decode(first_story_ids)

print(decoded)
print("-" * 50)
print("Last token of first story:", first_story_ids[-1])
print("Should equal eot_id:", eot_id)
print("Check passed:", first_story_ids[-1] == eot_id)