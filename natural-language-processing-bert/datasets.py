import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional, Union, Any
from d2l import torch as d2l


class SNLIDataset(Dataset):
    """用于加载SNLI数据集的自定义数据集"""

    """Stanford Natural Language Inference (SNLI) Dataset Handler

    This class processes and loads the SNLI dataset for natural language inference tasks,
    handling tokenization, padding, and label encoding.
    """

    def __init__(self,
                 dataset: Tuple[List[str], List[str], List[int]],
                 max_length: int,
                 vocab: Optional[d2l.Vocab] = None):
        """Initialize the SNLI dataset.

        Args:
            dataset: Tuple containing (premises, hypotheses, labels)
            max_length: Maximum sequence length for padding/truncation
            vocab: Predefined vocabulary, if None a new one will be created
        """
        self.max_length = max_length
        self.premises_raw, self.hypotheses_raw, self.labels_raw = dataset

        # Tokenize the text data
        self.premise_tokens = d2l.tokenize(self.premises_raw)
        self.hypothesis_tokens = d2l.tokenize(self.hypotheses_raw)
        if vocab is None:
            self.vocab = d2l.Vocab(self.premise_tokens + self.hypothesis_tokens,
                                   min_freq=5,
                                   reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab

        # Process and pad sequences
        self.premises = self._pad_sequences(self.premise_tokens)
        self.hypotheses = self._pad_sequences(self.hypothesis_tokens)
        self.labels = torch.tensor(self.labels_raw, dtype=torch.long)

        print('read ' + str(len(self.premises)) + ' examples from SNLI dataset')

    def _pad_sequences(self, tokenized_lines: List[List[str]]) -> torch.Tensor:
        """Pad or truncate sequences to max_length.

        Args:
            tokenized_lines: List of tokenized text sequences

        Returns:
            torch.Tensor: Padded sequences as tensor
        """
        return torch.tensor([
            d2l.truncate_pad(self.vocab[line],
                             self.max_length,
                             self.vocab['<pad>']
                            ) for line in tokenized_lines
        ])

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Get a single example by index.

        Args:
            index: Index of the example to retrieve

        Returns:
            Tuple containing ((premise, hypothesis), label)
        """
        return (self.premises[index], self.hypotheses[index]), self.labels[index]

    def __len__(self) -> int:
        """Get the total number of examples.

        Returns:
            int: Number of examples in the dataset
        """
        return len(self.premises)


def preprocess_text(text: str) -> str:
    """Clean and preprocess raw text data.

    Args:
        text: Raw text string

    Returns:
        str: Cleaned text string
    """
    # Remove parentheses
    text = re.sub(r'[()]', '', text)
    # Replace multiple whitespace with single space
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def load_snli_data(data_dir: str, is_train: bool) -> tuple[list[Any], list[Any], list[Any]]:
    """将SNLI数据集解析为前提、假设和标签"""
    """Load and parse SNLI dataset files.
    
    Args:
        data_dir: Directory containing SNLI dataset files
        is_training: Whether to load training or test data
        
    Returns:
        Tuple containing (premises, hypotheses, encoded_labels)
    """

    LABEL_MAPPING: Dict[str, int] = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    file_name = os.path.join(data_dir,
                             'snli_1.0_train.txt' if is_train else 'snli_1.0_test.txt')

    with open(file_name, 'r', encoding='utf-8') as f:
        # Skip header and split rows
        rows = [row.split('\t') for row in f.readlines()[1:]]

    # Extract and filter valid examples
    premises = []
    hypotheses = []
    labels = []

    for row in rows:
        if len(row) >= 3 and row[0] in LABEL_MAPPING:
            premises.append(preprocess_text(row[1]))
            hypotheses.append(preprocess_text(row[2]))
            labels.append(LABEL_MAPPING[row[0]])

    return premises, hypotheses, labels


def create_snli_dataloaders(batch_size: int,
                            max_length: int = 50,
                            num_workers: int = 0) -> Tuple[DataLoader, DataLoader, d2l.Vocab]:
    """下载SNLI数据集并返回数据迭代器和词表"""
    """Create data loaders for SNLI dataset training and evaluation.

    Args:
        batch_size: Size of each batch
        max_length: Maximum sequence length
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple containing (train_loader, test_loader, vocabulary)
    """

    num_workers = d2l.get_dataloader_workers()

    # Download and extract dataset
    data_dir = d2l.download_extract('SNLI')

    # Load training and test data
    train_data = load_snli_data(data_dir, True)
    test_data = load_snli_data(data_dir, False)

    # Create datasets
    train_set = SNLIDataset(train_data, max_length)
    test_set = SNLIDataset(test_data, max_length, vocab=train_set.vocab)

    # Create data loaders
    train_iter = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())

    test_iter = DataLoader(
        test_set,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())

    return train_iter, test_iter, train_set.vocab