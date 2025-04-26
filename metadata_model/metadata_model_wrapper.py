import torch
from metadata_model.metadata_model import SimpleNN

class metadata_model_wrapper:
    """
    A wrapper class for the SimpleNN model that should load a model from a file and provide methods for getting predictions and confidence scores.
    """
    
    def __init__(self, model_path: str = "../metadata_model/model.pth"):
        """
        Initializes the model wrapper by loading the model from the specified path.
        
        :param model_path: Path to the saved model file.
        """
        self.model = SimpleNN(input_size=4)  # Adjust input size as needed
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to evaluation mode

    def predict(self, x: torch.Tensor) -> bool:
        """
        Predicts the output for the given input tensor.
        
        :param x: Input tensor.
        :return: Predicted output tensor.
        """
        with torch.no_grad():
            return self.model(x).squeeze() >= 0.5
        
    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the confidence scores for the given input tensor.
        
        :param x: Input tensor.
        :return: Confidence scores tensor.
        """
        with torch.no_grad():
            output = self.model(x)
            confidence_scores = torch.sigmoid(output)
            return confidence_scores.squeeze()