import torch
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import spacy
from typing import List, Tuple

# Load spacy language models
try:
    spacy_en = spacy.load('en_core_web_sm')
    spacy_fr = spacy.load('fr_core_news_sm')
except IOError:
    print("Downloading language model for the spaCy tokenizer")
    from spacy.cli import download
    download('en_core_web_sm')
    download('fr_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
    spacy_fr = spacy.load('fr_core_news_sm')

# Tokenizers
tokenize_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenize_fr = get_tokenizer('spacy', language='fr_core_news_sm')

# Function to reverse tokens
def reverse_tokens(tokens):
    return list(reversed(tokens))

# Build vocabularies
def yield_tokens(data_iter, tokenizer, index, reverse=False):
    for from_to_tuple in data_iter:
        tokens = tokenizer(from_to_tuple[index])
        if reverse:
            tokens = reverse_tokens(tokens)
        yield tokens

SRC = build_vocab_from_iterator(yield_tokens(Multi30k(split='train'), tokenize_en, index=0, reverse=True), 
                                specials=['<unk>', '<pad>', '<sos>', '<eos>'], special_first=True)
TRG = build_vocab_from_iterator(yield_tokens(Multi30k(split='train'), tokenize_fr, index=1), 
                                specials=['<unk>', '<pad>', '<sos>', '<eos>'], special_first=True)

SRC.set_default_index(SRC['<unk>'])
TRG.set_default_index(TRG['<unk>'])

class TranslationDataset(Dataset):
    def __init__(self, split: str):
        self.data = list(Multi30k(split=split))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
    src_batch, tgt_batch = [], []
    for (raw_src, raw_tgt) in batch:
        src_tokens = reverse_tokens(tokenize_en(raw_src.rstrip("\n")))
        tgt_tokens = tokenize_fr(raw_tgt.rstrip("\n"))
        src_batch.append(torch.tensor([SRC[token] for token in ['<sos>'] + src_tokens + ['<eos>']], dtype=torch.long))
        tgt_batch.append(torch.tensor([TRG[token] for token in ['<sos>'] + tgt_tokens + ['<eos>']], dtype=torch.long))
    
    src_batch = pad_sequence(src_batch, padding_value=SRC['<pad>'], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=TRG['<pad>'], batch_first=True)
    return src_batch, tgt_batch

def get_data_loader(batch_size: int, split: str = 'train') -> DataLoader:
    dataset = TranslationDataset(split)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=(split == 'train'))

def get_vocab_sizes():
    return len(SRC), len(TRG)

# Export both SRC and TRG
__all__ = ['get_data_loader', 'get_vocab_sizes', 'SRC', 'TRG']