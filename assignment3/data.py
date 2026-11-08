"""Data split, tokenization, Dataset, and DataLoaders for next-token prediction in Assignment 3."""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CONTEXT_LENGTH = 256
END_OF_TEXT = "<|endoftext|>"


def split_stories(text_path, data_dir, val_fraction=0.05):
    """Split the stories into training and validation sets, and save them.

    args:
        text_path: path to the TinyStories text file
        data_dir: directory in which the two text files are saved
        val_fraction: fraction of the stories used for validation
    """
    ####################################
    # Your implementation starts here. #
    ####################################
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    stories = [s.strip() for s in text.split(END_OF_TEXT) if s.strip()]

    rng = np.random.default_rng(0)
    indices = np.arange(len(stories))
    rng.shuffle(indices)

    num_val = int(len(stories) * val_fraction)
    val_indices = set(indices[:num_val].tolist())

    train_stories = [stories[i] for i in range(len(stories)) if i not in val_indices]
    val_stories = [stories[i] for i in range(len(stories)) if i in val_indices]

    os.makedirs(data_dir, exist_ok=True)
    sep = f"\n{END_OF_TEXT}\n"
    with open(os.path.join(data_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.write(sep.join(train_stories) + sep)
    with open(os.path.join(data_dir, "val.txt"), "w", encoding="utf-8") as f:
        f.write(sep.join(val_stories) + sep)
    ##################################
    # Your implementation ends here. #
    ##################################


def tokenize_stories(text_path, tokenizer, tokens_path):
    """Convert all stories in the text file into token IDs and save them.

    args:
        text_path: path to a text file of stories
        tokenizer: a tokenizers.Tokenizer
        tokens_path: path of the .npy file to save
    """
    ####################################
    # Your implementation starts here. #
    ####################################
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    stories = [s.strip() for s in text.split(END_OF_TEXT) if s.strip()]
    eot_id = tokenizer.token_to_id(END_OF_TEXT)

    batch_size = 10000
    arrays = []
    for start in range(0, len(stories), batch_size):
        batch = stories[start:start + batch_size]
        encodings = tokenizer.encode_batch(batch)
        for encoding in encodings:
            arrays.append(np.array(encoding.ids, dtype=np.uint16))
            arrays.append(np.array([eot_id], dtype=np.uint16))

    tokens = np.concatenate(arrays)
    np.save(tokens_path, tokens)
    ##################################
    # Your implementation ends here. #
    ##################################


class TinyStoriesDataset(Dataset):
    """Return (input, target) windows of a token sequence."""

    def __init__(self, tokens, context_length=CONTEXT_LENGTH):
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        """Return the number of non-overlapping windows."""
        num_windows = None
        ####################################
        # Your implementation starts here. #
        ####################################
        num_windows = (len(self.tokens) - 1) // self.context_length
        ##################################
        # Your implementation ends here. #
        ##################################
        return num_windows

    def __getitem__(self, index):
        """Return the input and target tokens of window `index`.

        outputs:
            x: shape (context_length,), torch.long, the input tokens
            y: shape (context_length,), torch.long, the target tokens, i.e.,
               the input tokens shifted left by one position
        """
        x, y = None, None
        ####################################
        # Your implementation starts here. #
        ####################################
        start = index * self.context_length
        window = self.tokens[start:start + self.context_length + 1].astype(np.int64)
        x = torch.from_numpy(window[:-1]).long()
        y = torch.from_numpy(window[1:]).long()
        ##################################
        # Your implementation ends here. #
        ##################################
        return x, y


def get_loader(data_dir, split, batch_size, context_length=CONTEXT_LENGTH):
    """Create the DataLoader of one set.

    args:
        data_dir: directory containing train.npy and val.npy
        split: "train" or "val"
        batch_size: number of windows per batch
        context_length: number of input tokens in each window
    outputs:
        loader: a DataLoader that returns batches (x, y), each of shape
            (batch_size, context_length)
    """
    loader = None
    ####################################
    # Your implementation starts here. #
    ####################################
    tokens = np.load(os.path.join(data_dir, f"{split}.npy"))
    dataset = TinyStoriesDataset(tokens, context_length=context_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
    ##################################
    # Your implementation ends here. #
    ##################################
    return loader
