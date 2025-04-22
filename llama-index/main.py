from typing import Any
import pandas as pd

from SpamDetector import SpamDetector

def get_metadata(df: pd.DataFrame) -> dict[str, Any]:
    metadata = {
        'sender': df.get('sender', '')[:100],
        'date': df.get('date', '')[:30],
        'subject': df.get('subject', '')[:200],
        'label': df.get('label', '')[:50]
    }

    return metadata

if __name__ == "__main__":
    df: pd.DataFrame = pd.read_csv("../data/datasets/Nazario.csv")
    sample = df.sample(n=1)
    sample_body = sample["body"].iloc[0]
    print(f"Sample body: {sample_body}\n\n")

    detector = SpamDetector(
        spam_index_path="../data/index_spam",
        ham_index_path="../data/index_ham",
        top_k=2
    )

    classification, confidence = detector.classify(sample_body, metadata=get_metadata(sample),)
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Actual answer: {"spam" if sample['label'].item() == 1 else "ham"}")