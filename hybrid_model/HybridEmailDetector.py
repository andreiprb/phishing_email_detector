import pandas as pd

from rag_model.core.SpamDetector import SpamDetector
from transformer_model.transformer_model import EmailClassifier
from metadata_model.metadata_model_wrapper import MetadataModelWrapper
import config

from hybrid_model.utils import generate_rapport_body
from hybrid_model.models import RaportModel

class HybridEmailDetector:
    def __init__(self, confidence_treshold: float = 0.7):
        self.rag_model = SpamDetector(
            spam_index_path=config.INDEX_SPAM_PATH,
            ham_index_path=config.INDEX_HAM_PATH,
            top_k=1
        )
        self.transformer_model = EmailClassifier()
        self.metadata_model = MetadataModelWrapper()

        self.confidence_treshold = confidence_treshold
        self.raports: list[RaportModel] = []

    def get_last_raport(self):
        return self.raports[-1] if self.raports else None

    def get_raports(self):
        return self.raports

    def predict(self, subject: str, body: str, metadata: dict, make_raport: bool = False, use_rag: bool=False):
        prediction_model, confidence_model = self._model_analysys(subject, body, metadata)
        prediction_rag, confidence_rag = None, None

        if confidence_model < self.confidence_treshold and use_rag:
            prediction_rag, confidence_rag = self._rag_analysys(subject, body, metadata)

        if make_raport:
            raport = RaportModel(
                subject=subject,
                body=body,
                metadata=metadata,
                prediction_model="spam" if prediction_model else "ham",
                confidence_model=confidence_model,
                prediction_rag="spam" if prediction_rag else "ham",
                confidence_rag=confidence_rag
            )

            raport.raport_body = generate_rapport_body(raport)
            self.raports.append(raport)

        return (prediction_rag, confidence_rag) if prediction_rag else (prediction_model, confidence_model)

    def _model_analysys(self, subject: str, body: str, metadata: dict):
        """
        self.transformer_model.evaluate()
        self.metadata_model.evaluate()
        :return: prediction and confidence score
        """

        answer_trans = self.transformer_model.predict_email()
        answer_metad = self.metadata_model.predict()

        prediction, confidence = self._evaluate_models_answers(answer_trans, answer_metad)

        return prediction, confidence

    def _evaluate_models_answers(self,
        answer_trans: tuple[bool, float],
        answer_metad: tuple[bool, float],
        transformer_weight: float = .8) -> tuple[bool, float]:

        pred_meta, conf_meta = answer_metad
        pred_transformer, conf_transformer = answer_trans
        meta_weight = 1 - transformer_weight

        vote_meta = (conf_meta if pred_meta else -conf_meta) * meta_weight
        vote_transformer = (conf_transformer if pred_transformer else -conf_transformer) * transformer_weight

        combined_vote = vote_meta + vote_transformer

        final_pred = combined_vote > 0
        final_conf = min(abs(combined_vote), 1.0)

        return final_pred, final_conf

    def _rag_analysys(self, subject: str, body: str, metadata: dict):
        prediction, confidence = self.rag_model.classify()

        return prediction, confidence







