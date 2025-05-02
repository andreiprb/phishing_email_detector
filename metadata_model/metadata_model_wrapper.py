import pandas as pd
from typing import Dict, Tuple
import sys

import torch
from metadata_model.metadata_model import NotSimpleNN
from metadata_model.data_holder import string_to_int
import pandas as pd
from typing import Tuple
class MetadataModelWrapper:
    """
    A wrapper class for the SimpleNN model that should load a model from a file and provide methods for getting predictions and confidence scores.
    """
    
    def __init__(self, model_path: str):
        """
        Initializes the model wrapper by loading the model from the specified path.
        
        :param model_path: Path to the saved model file.
        """
        self.model = NotSimpleNN(input_size=3)  # Adjust input size as needed
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to evaluation mode

    def predict(self, x: torch.Tensor) -> bool:
        """
        Predicts the output for the given input tensor.
        true - spam, false - ham
        
        :param x: Input tensor.
        :return: Predicted output tensor.
        """
        with torch.no_grad():
            return self.model(x).squeeze() >= 0.5
        
    def _email_to_tensor(self, email: pd.DataFrame) -> torch.Tensor:
        """
        Converts an email object to a tensor representation.
        
        :param email: Email object to convert.
        :return: Tensor representation of the email.
        """
        sender: str = email.get('sender', '')
        date = email.get('date', '')
        urls = email.get('urls', [])
        # Convert the email features to a tensor representation
        x = torch.tensor([
            string_to_int(sender.values[0]),
            string_to_int(date.values[0]),
            urls.values[0],
        ], dtype=torch.float32)
        return x
        
    def get_confidence(self, x: torch.Tensor) -> float:
        """
        Computes the confidence scores for the given input tensor.
        
        :param x: Input tensor.
        :return: Confidence scores tensor.
        """
        with torch.no_grad():
            output = self.model(x)
            return output.softmax(dim=0).item()
        
    def get_prediction_and_confidence_from_email(self, email: pd.DataFrame) -> Tuple[bool, float]:
        """
        Computes the prediction and confidence score for the given input tensor.
        
        :param x: Input tensor.
        :return: Tuple of (prediction, confidence score).
        """
        with torch.no_grad():
            x = self._email_to_tensor(email)
            output = self.model(x)
            # Convert the output to a binary prediction (0 or 1) and confidence score
            prediction = output.squeeze() >= 0.5
            # Compute the confidence score using sigmoid activation
            confidence_score = self.get_confidence(x)

            return prediction, confidence_score
        
    def _create_tensor_from_dict(self, email: dict) -> torch.Tensor:
        """
        Converts a dictionary representation of an email to a tensor.
        
        :param email: Dictionary containing the email data.
        :return: Tensor representation of the email.
        """
        # Extract features from the dictionary and convert to tensor
        x = torch.tensor([
            string_to_int(email.get('sender', '')),
            string_to_int(email.get('date', '')),
            email.get('urls', ''),
        ], dtype=torch.float32)
        return x
    
    def get_prediction_from_dict(self, email: dict) -> bool:
        """
        Computes the prediction and confidence score for the given dictionary representation of an email.
        
        :param email: Dictionary containing the email data.
        :return: Tuple of (prediction, confidence score).
        """
        x = self._create_tensor_from_dict(email)
        prediction = self.predict(x)
        
        return prediction
        
    def test_model(self, test_data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Tests the model on a given test dataset and returns the accuracy.
        
        :param test_data: DataFrame containing the test dataset.
        :return: Accuracy of the model on the test dataset.
        """
        email = test_data.sample(n=1, random_state=42)
        prediction, confidence = self.get_prediction_and_confidence_from_email(email)
        return prediction, confidence
    
if __name__ == "__main__":
    # Example usage

    model_path = "data/models/model.pth"
    metadata_model_wrapper = MetadataModelWrapper(model_path)

    emails = pd.read_csv("data/datasets/CEAS_08.csv")
    test_data = emails.sample(n=1, random_state=44)

    prediction, confidence = metadata_model_wrapper.get_prediction_and_confidence_from_email(test_data)
    print(f"Prediction: {prediction}, Confidence: {confidence}")
    