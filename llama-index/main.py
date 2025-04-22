import pandas as pd

from SpamDetector import SpamDetector

if __name__ == "__main__":
    df = pd.read_csv("../data/datasets/Nazario.csv")
    sample = df.sample(n=1)
    sample_body = sample["body"].iloc[0]
    print(f"Sample body: {sample_body}\n\n")

    detector = SpamDetector(
        spam_index_path="../data/index_spam",
        ham_index_path="../data/index_ham",
        top_k=5
    )

    classification, confidence = detector.classify(sample_body)
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Actual answer: {"spam" if sample['label'].item() == 1 else "ham"}")