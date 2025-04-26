import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import hashlib
from typing import Tuple, List, Dict, Union

def string_to_int(s):
    # Use a hash function and map to a fixed integer space
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (10**8)  # 8-digit integer
class EmailDataset(Dataset):
    def __init__(self, path_csv: Union[str, List[str]], label_position: Union[str, List[str]] = None):
        """
        Initializes the dataset using path(s) to CSV file(s) and converts data to tensors
        
        :param path_csv: path to a CSV file or list of paths to CSV files
        :param label_position: position of the label column for each file
                            "second_last" (default) - label is the second to last column
                            "last" - label is the last column
                            Can be a single string (applies to all files) or a list matching path_csv
        """
        super().__init__()
        
        # Convert single path to list for uniform processing
        if isinstance(path_csv, str):
            path_csv = [path_csv]
            
        # Validate all paths
        for path in path_csv:
            print(f'Loading file: {path}')
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
        
        # Process all files and combine the data
        all_x_data = []
        all_y_data = []
        self.feature_maps = {}
        self.label_map = {}
        
        for idx, (path, pos) in enumerate(zip(path_csv, label_position)):
            # Load the CSV file
            data = pd.read_csv(path)
            
            # Get column names
            columns = list(data.columns)
            
            # Determine label column based on label_position
            if pos == "last":
                label_col = columns[-1]  # Last column
                feature_cols = columns[:-1]  # All except last column
            else:  # "second_last"
                label_col = columns[-2]  # Second to last column
                feature_cols = columns[:-2] + [columns[-1]]  # All except second to last column
            
            # Handle labels (y)
            if data[label_col].dtype == np.object_:
                # Create a mapping for categorical labels
                unique_labels = data[label_col].unique()
                current_map_size = len(self.label_map)
                
                # Add new labels to the map
                for label in unique_labels:
                    if label not in self.label_map:
                        self.label_map[label] = current_map_size
                        current_map_size += 1
                
                # Convert labels to numeric
                labels_numeric = np.array([self.label_map[label] for label in data[label_col]], dtype=np.float32)
                all_y_data.append(labels_numeric)
            else:
                # Labels are already numeric
                all_y_data.append(data[label_col].values.astype(np.float32))
            
            # Handle features (x)
            # For text/categorical columns, we'll need to encode them numerically
            x_numeric = []
            feature_cols = [col for col in feature_cols if col.lower() not in ['body', 'subject', 'receiver', 'label']]  # Exclude body and subject columns from features

            for col in feature_cols:
                col_data = data[col]
                # Check if column is numeric
                if np.issubdtype(col_data.dtype, np.number):
                    # If numeric, convert directly to numpy array
                    x_numeric.append(col_data.values.reshape(-1, 1))
                else:
                    # Convert to numeric using the mapping
                    encoded_values = np.array([string_to_int(value) for value in col_data], dtype=np.float32)
                    x_numeric.append(encoded_values.reshape(-1, 1))
            
            # Combine all feature columns for this file
            if x_numeric:
                x_combined = np.hstack(x_numeric)
                all_x_data.append(x_combined)
        
        # Combine all data into single tensors
        if all_x_data:
            x_combined = np.vstack(all_x_data)
            y_combined = np.concatenate(all_y_data)
            
            self.x = torch.tensor(x_combined, dtype=torch.float32)
            self.y = torch.tensor(y_combined, dtype=torch.float32)
        else:
            # Empty feature set
            self.x = torch.tensor([], dtype=torch.float32)
            self.y = torch.tensor([], dtype=torch.float32)
        
        # Store file boundaries for reference
        self.file_boundaries = []
        cumulative_size = 0
        for x_array in all_x_data:
            cumulative_size += x_array.shape[0]
            self.file_boundaries.append(cumulative_size)

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
                *[self.file_boundaries[i] - self.file_boundaries[i-1] for i in range(1, len(self.file_boundaries))]
            ]
        }
    
    def __len__(self):
        """
        Returns the length of the dataset
        :return: length of the dataset
        """
        return len(self.y)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the features and label from position :param index:
        :param index: position from which to return the features and its label
        :return: tuple: (features as tensor, label as tensor)
        """
        return self.x[index], self.y[index]