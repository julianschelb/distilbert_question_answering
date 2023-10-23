# %% [markdown]
# # Load Data
# This notebook loads and preproceses all necessary data, namely the following.
# * OpenWebTextCorpus: for base DistilBERT model
# * SQuAD datasrt: for Q&A
# * Natural Questions (needs to be downloaded externally but is preprocessed here): for Q&A
# * HotPotQA: for Q&A

# %%
from tqdm.auto import tqdm
from datasets import load_dataset
import os
import pandas as pd
import random

# %% [markdown]
# ## Distilbert Data
# In the following, we download the english openwebtext dataset from huggingface (https://huggingface.co/datasets/openwebtext). The dataset is provided by Aaron Gokaslan and Vanya Cohen from Brown University (https://skylion007.github.io/OpenWebTextCorpus/).
# 
# We first load the data, investigate the structure and write the dataset into files of each 10 000 texts.

# %%
ds = load_dataset("openwebtext")

# %%
# we have a text-only training dataset with 8 million entries
ds

# %%
# create necessary folders
os.mkdir('data')
os.mkdir('data/original')

# %%
# save text in chunks of 10000 samples
text = []
ind = 0

for sample in tqdm(ds['train']):
    # replace all newlines
    sample = sample['text'].replace('\n','')
    
    # append cleaned sample to all texts
    text.append(sample)
    
    # if we processed 10000 samples, write them to a file and start over
    if len(text) == 10000:
        with open(f"data/original/text_{ind}.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(text))
        text = []
        ind += 1

# write remaining samples to a file
with open(f"data/original/text_{ind}.txt", 'w', encoding='utf-8') as f:
    f.write('\\n'.join(text))

# %% [markdown]
# ### Testing
# If we load the first file, we should get a file that is 10000 lines long and has one column
# 
# As we do not preprocess the data in any way, but just write the read text into the file, this is all testing necessary

# %%
with open("data/original/text_0.txt", 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
lines = pd.DataFrame(lines)

# %%
assert lines.shape==(10000,1)
print("Passed")


