import os
import pandas as pd
from pathlib import Path
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode

def load_sample(file_path: Path, sample_size: Optional[int] = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Safely load and validate CSV file, with proper error handling.

    Args:
        file_path: Path to the CSV file
        sample_size: Number of samples to take from each category (if available)

    Returns:
        Tuple of (spam_sample, ham_sample) DataFrames
    """

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {file_path}")

    try:
        # First read with pandas for validation and preprocessing
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")

        # Count samples in each category
        spam_count = (df["label"] == 1).sum()
        ham_count = (df["label"] == 0).sum()

        # Take minimum of requested sample size and available samples
        spam_sample_size = min(sample_size, spam_count)
        ham_sample_size = min(sample_size, ham_count)

        # Sample with adjusted sizes
        spam_sample = df[df["label"] == 1].sample(spam_sample_size) if spam_sample_size > 0 else pd.DataFrame()
        ham_sample = df[df["label"] == 0].sample(ham_sample_size) if ham_sample_size > 0 else pd.DataFrame()

        return spam_sample, ham_sample

    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        raise

def preprocess_email_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess email data for better indexing in LlamaIndex

    Args:
        df: DataFrame with raw email data

    Returns:
        Preprocessed DataFrame
    """
    logger.info("Preprocessing email data...")

    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Fill empty fields with empty strings to avoid NaN issues
    processed_df = processed_df.fillna("")

    # Create a combined text field that LlamaIndex can use more effectively
    processed_df['combined_text'] = processed_df.apply(
        lambda row: (
            f"From: {row['sender']}\n"
            f"To: {row['receiver']}\n"
            f"Date: {row['date']}\n"
            f"Subject: {row['subject']}\n\n"
            f"{row['body']}"
        ), axis=1
    )

    logger.info("Preprocessing complete")
    return processed_df

def create_documents_from_dataframe(df: pd.DataFrame) -> List[Document]:
    """
    Convert DataFrame rows to LlamaIndex Document objects

    Args:
        df: Preprocessed DataFrame

    Returns:
        List of LlamaIndex Document objects
    """
    logger.info("Creating LlamaIndex documents from DataFrame...")

    documents = []

    for idx, row in df.iterrows():
        # Create metadata dictionary with only essential fields
        metadata = {
            'sender': str(row['sender'])[:100],
            'receiver': str(row['receiver'])[:100],
            'date': str(row['date'])[:30],
            'subject': str(row['subject'])[:200],
            'label': str(row.get('label', ''))[:50],
            'email_id': f"email_{idx}"
        }

        # Create a Document with the combined text and metadata
        doc = Document(
            text=row['combined_text'],
            metadata=metadata,
            id_=f"email_{idx}"
        )
        documents.append(doc)

    logger.info(f"Created {len(documents)} documents")
    return documents

def chunk_documents(documents: List[Document], chunk_size: int = 8192, chunk_overlap: int = 20) -> List[Document]:
    """
    Chunk documents into smaller nodes for better indexing

    Args:
        documents: List of documents to chunk
        chunk_size: Size of each chunk (increased to handle large metadata)
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked documents
    """
    logger.info(f"Chunking documents with size={chunk_size}, overlap={chunk_overlap}")

    # Extract only essential metadata to reduce size
    for doc in documents:
        # Keep only essential metadata and limit their size
        essential_metadata = {
            'sender': doc.metadata.get('sender', '')[:100],
            'date': doc.metadata.get('date', '')[:30],
            'subject': doc.metadata.get('subject', '')[:200],
            'label': doc.metadata.get('label', '')[:50]
        }
        # Replace the original metadata with the trimmed version
        doc.metadata = essential_metadata

    parser = SimpleNodeParser.from_defaults(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    nodes = parser.get_nodes_from_documents(documents)
    logger.info(f"Created {len(nodes)} nodes from {len(documents)} documents")

    return nodes

def create_index_from_documents(nodes: List[TextNode]) -> VectorStoreIndex:
    """
    Create a LlamaIndex VectorStoreIndex from documents

    Args:
        nodes: List of nodes to index

    Returns:
        Initialized VectorStoreIndex
    """
    logger.info("Creating vector index...")

    index: VectorStoreIndex = VectorStoreIndex(nodes)
    logger.info("Vector index created successfully")

    return index