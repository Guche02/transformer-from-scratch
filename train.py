# !pip install datasets
# !pip install tokenizer

import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path  # it is a library used to create absolute path from relative path
from torch.utils.data import Dataset, DataLoader, random_split

"""yield: Pauses the function and saves its state so it can be resumed later. Multiple values can be yielded over time.

 return: Ends the function entirely, returning a single value or object.

Example of dataset:

ds = [
    {"translation": {"en": "Hello", "fr": "Bonjour"}},
    {"translation": {"en": "Goodbye", "fr": "Au revoir"}},
    {"translation": {"en": "Thank you", "fr": "Merci"}}
]

"""

def get_all_sentences(ds, lang):
  for item in ds:
    yield item['translation'][lang]    # it iterates through the ds and returns an item for a defined language

def get_or_build_tokenizer(config, ds, lang):
  tokenizer_path = Path(config['tokenizer_file'].format(lang))
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))                                                # if unknown words from the vocab are seen, replace it my [UNK] token
    tokenizer.pre_tokenizer = Whitespace()                                                             # break the words into tokens by whitespace
    trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)   # min_freq = 2 means if the word occurs more than 2 times then add it to the vocab
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)                        # get_all_sentences is used to get all the sentences from the library
    tokenizer.save(str(tokenizer_path))
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))                      # if the tokenizer already exists, then simply load it from the file path
  return tokenizer

def get_ds(config):
  ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

  # building the source and target tokenizers
  tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
  tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

  # train_val split
  train_ds_size = int(0.9 * len(ds_raw))      # 90% in the train dataset
  val_ds_size = len(ds_raw) - train_ds_size   # remaining in the val dataset

  train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])   # random split is similar to train_test_split in pytorch