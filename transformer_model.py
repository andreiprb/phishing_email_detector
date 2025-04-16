import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import re
import os
from typing import Tuple, List, Dict, Union
from collections import Counter


class EmailDataset(Dataset):
    def __init__(self, path_csv: Union[str, List[str]], label_position: Union[str, List[str]] = None, max_len=200):
        """
        Initializes the dataset using path(s) to CSV file(s) and converts data to tensors

        :param path_csv: path to a CSV file or list of paths to CSV files
        :param label_position: position of the label column for each file
                            "second_last" (default) - label is the second to last column
                            "last" - label is the last column
                            Can be a single string (applies to all files) or a list matching path_csv
        :param max_len: maximum sequence length for tokenization
        """
        super().__init__()
        self.max_len = max_len

        # Convert single path to list for uniform processing
        if isinstance(path_csv, str):
            path_csv = [path_csv]

        # Validate all paths
        for path in path_csv:
            assert os.path.exists(path) and os.path.isfile(path), f'The path {path} does not point to a file'

        # Set default label positions if not provided
        if label_position is None:
            label_position = ["second_last"] * len(path_csv)
        # If a single position is provided, apply to all files
        elif isinstance(label_position, str):
            label_position = [label_position] * len(path_csv)
        # Ensure label_position matches the number of files
        assert len(label_position) == len(path_csv), "Number of label positions must match number of files"

        # Store for reference
        self.paths_csv = path_csv
        self.label_positions = label_position

        # Process all files and extract data
        all_texts = []
        all_labels = []
        self.file_boundaries = []

        for idx, (path, pos) in enumerate(zip(path_csv, label_position)):
            # Load the CSV file
            data = pd.read_csv(path)

            # Get column names
            columns = list(data.columns)

            # Determine label column based on label_position
            if pos == "last":
                label_col = columns[-1]  # Last column
                # Assume subject and body columns exist (adjust as needed)
                if len(columns) >= 3:
                    subject_col = columns[0]
                    body_col = columns[1]
                else:
                    subject_col = columns[0]
                    body_col = None
            else:  # "second_last"
                label_col = columns[-2]  # Second to last column
                # Assume subject and body columns exist (adjust as needed)
                if len(columns) >= 3:
                    subject_col = columns[0]
                    body_col = columns[1]
                else:
                    subject_col = columns[0]
                    body_col = None

            # Create text field by combining subject and body
            if subject_col and body_col and body_col in columns:
                data['text'] = data[subject_col].astype(str) + " " + data[body_col].astype(str)
            elif subject_col:
                data['text'] = data[subject_col].astype(str)
            else:
                # Fallback if no subject/body columns found
                data['text'] = "No text available"

            # Preprocess text
            data['text'] = data['text'].apply(self.preprocess_text)

            # Process labels
            if data[label_col].dtype == object:
                # Convert categorical labels to binary (assuming spam/ham classification)
                # This is a simplification; adjust based on your actual labels
                spam_indicators = ['spam', '1', 'true', 'yes', 'positive']
                labels_numeric = np.array([1.0 if str(label).lower() in spam_indicators else 0.0
                                           for label in data[label_col]], dtype=np.float32)
            else:
                # Numeric labels are used as is
                labels_numeric = data[label_col].values.astype(np.float32)

            # Collect texts and labels
            texts_for_file = data['text'].values
            all_texts.extend(texts_for_file)
            all_labels.extend(labels_numeric)

            # Track file boundaries
            current_boundary = len(all_texts)
            self.file_boundaries.append(current_boundary)

        # Build vocabulary
        self.vocab = self.build_vocab(all_texts, 20000)  # 20000 max features

        # Store data
        self.texts = all_texts
        self.labels = np.array(all_labels, dtype=np.float32)

    def preprocess_text(self, text):
        """
        Preprocess text by converting to lowercase and removing non-alphanumeric characters

        :param text: Text to preprocess
        :return: Preprocessed text
        """
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text

    def build_vocab(self, texts, max_tokens):
        """
        Build vocabulary from a list of texts

        :param texts: List of texts
        :param max_tokens: Maximum vocabulary size
        :return: Dictionary mapping words to indices
        """
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        # Get most common words
        common_words = [word for word, _ in counter.most_common(max_tokens - 2)]  # -2 for <PAD> and <UNK>

        # Create a word-to-index mapping
        word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        for i, word in enumerate(common_words):
            word_to_idx[word] = i + 2

        return word_to_idx

    def get_file_index(self, index: int) -> int:
        """
        Returns the file index that contains the record at the given index

        :param index: dataset index
        :return: file index (0-based)
        """
        for i, boundary in enumerate(self.file_boundaries):
            if index < boundary:
                return i
        return len(self.file_boundaries) - 1  # Last file

    def get_file_info(self) -> Dict:
        """
        Returns information about the loaded files and their label positions

        :return: Dictionary with file paths and label positions
        """
        return {
            "files": self.paths_csv,
            "label_positions": self.label_positions,
            "samples_per_file": [
                self.file_boundaries[0] if self.file_boundaries else 0,
                *[self.file_boundaries[i] - self.file_boundaries[i - 1] for i in range(1, len(self.file_boundaries))]
            ]
        }

    def __len__(self):
        """
        Returns the length of the dataset
        :return: length of the dataset
        """
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the tokenized text and label at position index
        :param index: position from which to return the data
        :return: tuple: (tokenized text as tensor, label as tensor)
        """
        text = self.texts[index]
        label = self.labels[index]

        # Tokenize the text
        tokens = text.split()
        # Convert tokens to indices
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        # Pad or truncate to max_len
        if len(indices) < self.max_len:
            indices += [self.vocab["<PAD>"]] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)


def create_data_loaders(emails, label_positions, batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                        max_len=200, random_seed=42):
    """
    Create train, validation, and test data loaders from email files

    :param emails: List of email CSV files
    :param label_positions: List of label positions for each file
    :param batch_size: Batch size for loaders
    :param train_ratio: Ratio of data for training
    :param val_ratio: Ratio of data for validation
    :param test_ratio: Ratio of data for testing
    :param max_len: Maximum sequence length
    :param random_seed: Random seed for reproducibility
    :return: Tuple of (train_loader, val_loader, test_loader, dataset)
    """
    # Set random seed
    torch.manual_seed(random_seed)

    # Create dataset
    print(f"Loading data from {len(emails)} CSV files...")
    dataset = EmailDataset(emails, label_positions, max_len=max_len)
    print(f"Dataset loaded with {len(dataset)} samples")

    # Print file information
    file_info = dataset.get_file_info()
    for i, (file, samples) in enumerate(zip(file_info['files'], file_info['samples_per_file'])):
        print(f"  {file}: {samples} samples, label position: {file_info['label_positions'][i]}")

    # Split dataset
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, dataset


# Define the TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x):
        # PyTorch's MultiheadAttention expects input shape (seq_len, batch, embed_dim)
        # But our input is (batch, seq_len, embed_dim)
        attn_input = x.permute(1, 0, 2)

        # Self-attention block
        attn_output, _ = self.att(attn_input, attn_input, attn_input)

        # Convert back to (batch, seq_len, embed_dim)
        attn_output = attn_output.permute(1, 0, 2)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        # Feed forward block
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)

        return self.layernorm2(out1 + ffn_output)


# Define the full model
class EmailTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, ff_dim, max_len, num_transformer_blocks):
        super(EmailTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim, num_heads, ff_dim) for _ in range(num_transformer_blocks)]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(embedding_dim, 20)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.embedding(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Global average pooling
        # Change shape from (batch, seq_len, features) to (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x).squeeze(-1)

        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc2(x))

        return x


# Utility functions for training and evaluation
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return total_loss / len(train_loader), accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return total_loss / len(val_loader), accuracy


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return total_loss / len(test_loader), accuracy


def predict_email(model, tokenized_input, device):
    """
    Predicts whether an email is spam or not

    :param model: The trained transformer model
    :param tokenized_input: Tokenized and padded email text (tensor)
    :param device: The device to run inference on
    :return: Tuple of (prediction, confidence)
    """
    # Make sure input is a tensor and add batch dimension if needed
    if not isinstance(tokenized_input, torch.Tensor):
        raise TypeError("Input must be a tensor")

    if len(tokenized_input.shape) == 1:
        tokenized_input = tokenized_input.unsqueeze(0)

    # Move to device
    tokenized_input = tokenized_input.to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(tokenized_input).squeeze()
        prediction = (output > 0.5).float().item()

    return prediction, output.item()


def train_model(
        model,
        train_loader,
        val_loader,
        test_loader=None,
        epochs=5,
        lr=0.001,
        device=None
):
    """
    Train the transformer model

    :param model: The EmailTransformer model
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data
    :param test_loader: Optional DataLoader for test data
    :param epochs: Number of training epochs
    :param lr: Learning rate
    :param device: Device to train on (will use CUDA if available when None)
    :return: Trained model and training history
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device("cuda") if torch.cuda.is_available() else device

    print(f"Using device: {device}")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

    # Evaluate on test set if provided
    if test_loader:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        history['test_loss'] = test_loss
        history['test_acc'] = test_acc

    return model, history


def save_model(model, vocab, path="transformer_model.pth", extra_data=None):
    """
    Save the model, vocabulary, and any extra data

    :param model: Trained model to save
    :param vocab: Vocabulary (word_to_idx mapping)
    :param path: Path to save the model
    :param extra_data: Any additional data to save with the model
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab
    }

    if extra_data:
        save_dict.update(extra_data)

    torch.save(save_dict, path)
    print(f"Model saved to {path}")


def load_model(path, device=None):
    """
    Load a saved model

    :param path: Path to the saved model
    :param device: Device to load the model to
    :return: model, vocabulary, and any extra saved data
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device("cuda") if torch.cuda.is_available() else device

    # Load the saved dictionary
    saved_dict = torch.load(path, map_location=device)

    # Extract model parameters and vocabulary
    model_state_dict = saved_dict.pop('model_state_dict')
    vocab = saved_dict.pop('vocab')

    # Get model architecture parameters if they were saved
    model_params = saved_dict.pop('model_params', {
        'vocab_size': len(vocab),
        'embedding_dim': 128,
        'num_heads': 4,
        'ff_dim': 128,
        'max_len': 200,
        'num_transformer_blocks': 2
    })

    # Create model with saved parameters
    model = EmailTransformer(**model_params)
    model.load_state_dict(model_state_dict)
    model.to(device)

    # Return the model, vocabulary, and any remaining data
    return model, vocab, saved_dict


def main():
    # Path to the saved model
    model_path = "transformer_model.pth"

    # List of email datasets and their label positions
    emails = [
        "dataSources/CEAS_08.csv",
        "dataSources/Nazario_5.csv",
        "dataSources/Nazario.csv",
        "dataSources/Nigerian_5.csv",
        "dataSources/Nigerian_Fraud.csv",
        "dataSources/SpamAssasin.csv"
    ]

    label_positions = [
        "second_last",
        "second_last",
        "last",
        "second_last",
        "last",
        "second_last"
    ]

    # Create data loaders
    train_loader, val_loader, test_loader, dataset = create_data_loaders(
        emails,
        label_positions,
        batch_size=32,
        max_len=200
    )

    # Define model parameters
    vocab_size = len(dataset.vocab)
    embedding_dim = 128
    num_heads = 4
    ff_dim = 128
    max_len = 200
    num_transformer_blocks = 2

    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}...")
        trained_model, vocab, extra_data = load_model(model_path)
        print("Model loaded successfully.")
    else:
        print(f"Creating model with vocabulary size: {vocab_size}")

        # Create model
        model = EmailTransformer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_len=max_len,
            num_transformer_blocks=num_transformer_blocks
        )

        # Train model
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=5
        )

        # Save model
        save_model(
            trained_model,
            dataset.vocab,
            path=model_path,
            extra_data={
                'model_params': {
                    'vocab_size': vocab_size,
                    'embedding_dim': embedding_dim,
                    'num_heads': num_heads,
                    'ff_dim': ff_dim,
                    'max_len': max_len,
                    'num_transformer_blocks': num_transformer_blocks
                },
                'training_history': history
            }
        )
        print(f"Model training complete and saved to {model_path}")

    # Example of prediction
    print("\nExample prediction:")
    example_text = "Dear friend, I have an opportunity for you to make money. Please send your bank details."
    # Preprocess and tokenize like in the dataset
    example_text = example_text.lower()
    example_text = re.sub(r"[^a-z0-9\s]", "", example_text)

    # Tokenize manually
    tokens = example_text.split()
    indices = [dataset.vocab.get(token, dataset.vocab["<UNK>"]) for token in tokens]
    if len(indices) < max_len:
        indices += [dataset.vocab["<PAD>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    example_tensor = torch.tensor(indices, dtype=torch.long)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else device

    prediction, confidence = predict_email(trained_model, example_tensor, device)
    print(f"Text: {example_text}")
    print(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()