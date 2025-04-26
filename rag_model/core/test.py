from tqdm import tqdm
from typing import Any
import pandas as pd

import sys
sys.path.append("")

from SpamDetector import SpamDetector

def get_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """
    Extracts specific metadata information from the provided DataFrame.

    This function retrieves values associated with certain predefined keys
    ('sender', 'date', 'subject', and 'label') from the input DataFrame.
    If a key is not found within the DataFrame, the respective field in
    the metadata dictionary is defaulted to an empty string. Additionally,
    the values are capped to specific lengths where applicable:
    - 'sender': 100 characters
    - 'date': 30 characters
    - 'subject': 200 characters

    :param df: A pandas DataFrame containing the data from which metadata fields
        'sender', 'date', 'subject', and 'label' are extracted. Missing keys will
        result in empty string defaults.
    :type df: pd.DataFrame
    :return: A dictionary containing the metadata information with capped lengths
        for certain fields ('sender', 'date', 'subject').
    :rtype: dict[str, Any]
    """
    metadata = {
        'sender': df.get('sender', '')[:100],
        'date': df.get('date', '')[:30],
        'subject': df.get('subject', '')[:200],
        'label': df.get('label', '')
    }

    return metadata

def test_random_sample(detector: SpamDetector, df: pd.DataFrame):
    """
    Tests a random sample from the provided DataFrame to evaluate the detector's classification
    performance. It selects a single random sample, classifies it, and compares the result
    with the expected label to determine correctness.

    :param detector: The spam detector instance to classify the sample.
    :type detector: SpamDetector
    :param df: A pandas DataFrame containing the dataset with at least 'body' and 'label' columns.
    :type df: pd.DataFrame
    :return: A tuple containing the classification label, confidence score, and a boolean indicating
             whether the classification matches the expected label.
    :rtype: tuple
    """
    sample = df.sample(n=1, random_state=42)

    classification, confidence = detector.classify(sample['body'].iloc[0], metadata=get_metadata(sample))

    return classification, confidence, True if sample['label'].item() == classification else False

def test_accuracy(detector: SpamDetector, df: pd.DataFrame, tests_count: int = 20):
    """
    Tests the accuracy of a given spam detector by evaluating a sample of messages
    against their correct classifications. The function samples a subset of the
    dataframe, runs classifications using the detector, and calculates the
    percentage of correct classifications.

    :param detector: The spam detector object to be evaluated.
    :type detector: SpamDetector
    :param df: The dataframe containing sample data, where messages are stored in
        the 'body' column and their corresponding labels in the 'label' column.
    :type df: pd.DataFrame
    :param tests_count: The number of samples to be selected from the dataframe
        for accuracy testing. Default value is 20.
    :type tests_count: int
    :return: None
    """
    samples = df.sample(n=tests_count, random_state=42)

    correct_count = 0
    for _, sample in tqdm(samples.iterrows(), total=len(df)):
        classification, confidence = detector.classify(sample['body'], metadata=get_metadata(sample))
        if classification == sample['label'].item():
            correct_count += 1

    accuracy = correct_count / len(df)
    print(f"Accuracy: {accuracy:.2f}")