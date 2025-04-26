from rag_model.core.SpamDetector import SpamDetector
from transformer_model.transformer_model import EmailClassifier
from metadata_model.metadata_model_wrapper import MetadataModelWrapper

import config

class HybridEmailDetector:
    def __init__(self):
        self.rag_model = SpamDetector(
            spam_index_path=config.INDEX_SPAM_PATH,
            ham_index_path=config.INDEX_HAM_PATH,
            top_k=1
        )

        self.transformer_model = EmailClassifier()

        self.metadata_model = MetadataModelWrapper()