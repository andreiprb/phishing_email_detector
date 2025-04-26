from tqdm import tqdm
from typing import Any
import pandas as pd

from SpamDetector import SpamDetector

def get_metadata(df: pd.DataFrame) -> dict[str, Any]:
    metadata = {
        'sender': df.get('sender', '')[:100],
        'date': df.get('date', '')[:30],
        'subject': df.get('subject', '')[:200],
        'label': df.get('label', '')
    }

    return metadata

def test_random_sample(detector: SpamDetector, df: pd.DataFrame):
    sample = df.sample(n=1, random_state=42)

    classification, confidence = detector.classify(sample['body'].iloc[0], metadata=get_metadata(sample))

    return classification, confidence, True if sample['label'].item() == classification else False

def test_accuracy(detector: SpamDetector, df: pd.DataFrame, tests_count: int = 20):
    samples = df.sample(n=tests_count, random_state=42)

    correct_count = 0
    for _, sample in tqdm(samples.iterrows(), total=len(df)):
        classification, confidence = detector.classify(sample['body'], metadata=get_metadata(sample))
        if classification == sample['label'].item():
            correct_count += 1

    accuracy = correct_count / len(df)
    print(f"Accuracy: {accuracy:.2f}")