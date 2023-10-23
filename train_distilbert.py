# %% [markdown]
# # DistilBERT Base Model
# The following contains the code to create and train a DistilBERT model using the Huggingface library. It works quite well for a moderate amount of data, but the runtime increases quite drastically with data.
# 
# I decided to take the pretrained model after all, still, creating the model myself was quite interesting!

# %%
from pathlib import Path
import torch
import time
from pathlib import Path
from transformers import DistilBertTokenizerFast
import os
from transformers import DistilBertConfig
from transformers import DistilBertForMaskedLM
from tokenizers import BertWordPieceTokenizer
from tqdm.auto import tqdm
from torch.optim import AdamW
import torchtest
from transformers import pipeline


from distilbert import test_model
from distilbert import Dataset

import numpy as np

# %% [markdown]
# ## Tokeniser
# We need a way to convert the strings we get as the input to numerical tokens, that we can give to the neual network. Hence, we take a BertWorkPieceTokenizer (works for DistilBERT too) and create tokens from our words.

# %%
fit_new_tokenizer = True

if fit_new_tokenizer:
    paths = [str(x) for x in Path('data/original').glob('**/*.txt')]

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
    )

# %%
# fit the tokenizer
if fit_new_tokenizer:
    tokenizer.train(files=paths[:10], vocab_size=30_000, min_frequency=2,
                    limit_alphabet=1000, wordpieces_prefix='##',
                    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])

# %%
if fit_new_tokenizer:
    os.mkdir('./tokeniser')
    tokenizer.save_model('tokeniser')

# %% [markdown]
# After having created a basic tokeniser, we use the model to initialise a DistilBert tokenizer, that we need for the model architecture later on. We save the tokeniser separately.

# %%
tokenizer = DistilBertTokenizerFast.from_pretrained('tokeniser', max_len=512)
tokenizer.save_pretrained("distilbert_tokenizer")

# %% [markdown]
# ### Testing
# We now test the created tokenizer. We take a simple example and tokenise the input. It can be seen that we add a special token in the beginning and end ('CLS' and 'SEP'), which is how the BERT model was defined.
# 
# When we translate the input back, we can see that we get the same, except for the first and last token. Also, we can see that questionmarks and commas are encoded separately.

# %%
tokens = tokenizer('Hello, how are you?')
print(tokens)

# %%
tokenizer.decode(tokens['input_ids'])

# %%
for tok in tokens['input_ids']:
    print(tokenizer.decode(tok))

# %%
assert len(tokenizer.vocab) == 30_000

# %% [markdown]
# ## Dataset
# We now define a function to mask some of the tokens. In particular, we create a Dataset class, that automates loading the data and tokenising it for us. Lastly, we use a DataLoader to load the data step by step into memory.
# 
# The big problem with the limited resources we have is memory. In particular, I am loading the data sequentially, file by file, keeping track how many samples have been read. Shuffling wouldn't work here (it would also not make a lot of sense for this dataset).

# %%
# create dataset and dataloader 
dataset = Dataset(paths = [str(x) for x in Path('data/original').glob('**/*.txt')][50:70], tokenizer=tokenizer)
loader = torch.utils.data.DataLoader(dataset, batch_size=8)

test_dataset = Dataset(paths = [str(x) for x in Path('data/original').glob('**/*.txt')][10:12], tokenizer=tokenizer)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)

# %% [markdown]
# ### Testing
# The randomisation makes it a bit difficult to test. But altogether, we see that the input ids, masks and labels have the same shape. Also, as we mask 15% of the samples, when decoding a given sample, we can see that some samples are now '[MASK]'.

# %%
i = iter(dataset)

# %%
for j in range(10):
    sample = next(i)
    
    input_ids = sample['input_ids']
    attention_masks = sample['attention_mask']
    labels = sample['labels']
    
    # check if the dimensions are right
    assert input_ids.shape[0] == (512)
    assert attention_masks.shape[0] == (512)
    assert labels.shape[0] == (512)
    
    # if the input ids are not masked, the labels are the same as the input ids
    assert np.array_equal(input_ids[input_ids != 4].numpy(),labels[input_ids != 4].numpy())
    # input ids are zero if the attention masks are zero
    assert np.all(input_ids[attention_masks == 0].numpy()==0)
    # check if input contains masked tokens (we can't guarantee this 100% but this will apply) most likely
    assert np.any(input_ids.numpy() == 4)
print("Passed")

# %% [markdown]
# ## Model
# In the following section, we intialise and train a model.

# %%
config = DistilBertConfig(
    vocab_size=30000,
    max_position_embeddings=514
)

# %%
model = DistilBertForMaskedLM(config)

# %%
# if we have a GPU - train on gpu
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# %%
device

# %% [markdown]
# ### Testing the model
# I stumbled across some Medium articles on how to test DeepLearning models beforehand 
# * https://thenerdstation.medium.com/how-to-unit-test-machine-learning-code-57cf6fd81765: the package is however deprecated
# * https://towardsdatascience.com/testing-your-pytorch-models-with-torcheck-cb689ecbc08c: released a package (torcheck)
# * https://github.com/suriyadeepan/torchtest: I found this package, which is the PyTorch version of the first one and is still maintained.
# 
# Essentially, testing a model is inherently difficult, because we do not know the result in the beginning. Still, the following four conditions should be satisfied in every model (see second reference above):
# 1. The parameters should change during training (if they are not frozen).
# 2. The parameters should not change if they are frozen.
# 3. The range of the ouput should be in a predefined range.
# 4. The parameters should never contain NaN. The same goes for the outputs too.
# 
# I tried using the packages, but they do not trivially apply for models with multiple inputs (we have input ids and attention masks). The following is partly adapted from the torchtest package (https://github.com/suriyadeepan/torchtest/blob/master/torchtest/torchtest.py).

# %%
# get smaller dataset
test_ds = Dataset(paths = [str(x) for x in Path('data/original').glob('**/*.txt')][:2], tokenizer=tokenizer)
test_ds_loader = torch.utils.data.DataLoader(test_ds, batch_size=2)
optim=torch.optim.Adam(model.parameters())

# %%
from distilbert import test_model

test_model(model, optim, test_ds_loader, device)

# %% [markdown]
# ### Training the model
# We use AdamW as the optimiser and train for 10 epochs.
# 
# Taking the whole dataset, takes about 100 hours per epoch for me, so I wasn't able to do that.

# %%
model = DistilBertForMaskedLM(config)
# if we have a GPU - train on gpu
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# %%
# we use AdamW as the optimiser
optim = AdamW(model.parameters(), lr=1e-4)

# %%
epochs = 1

for epoch in range(epochs):
    loop = tqdm(loader, leave=True)
    
    # set model to training mode
    model.train()
    losses = []
    
    # iterate over dataset
    for batch in loop:
        optim.zero_grad()
        
        # copy input to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # predict
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # update weights
        loss = outputs.loss
        loss.backward()
        
        optim.step()
        
        # output current loss
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())
        
        del input_ids
        del attention_mask
        del labels
        
    print("Mean Training Loss", np.mean(losses))
    losses = []
    loop = tqdm(test_loader, leave=True)
    
    # set model to evaluation mode
    model.eval()
    
    # iterate over dataset
    for batch in loop:
        # copy input to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # predict
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # update weights
        loss = outputs.loss
        
        # output current loss
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())
        
        del input_ids
        del attention_mask
        del labels
    print("Mean Test Loss", np.mean(losses))

# %%
# save the pretrained model
torch.save(model, "distilbert.model")

# %%
model = torch.load("distilbert.model")

# %% [markdown]
# ### Testing
# Huggingface provides a library to quickly be able to see what word the model would predict for our masked token.

# %%
fill = pipeline("fill-mask", model='distilbert', config=config, tokenizer='distilbert_tokenizer')

# %%
fill(f'It seems important to tackle the climate {fill.tokenizer.mask_token}.')

# %%



