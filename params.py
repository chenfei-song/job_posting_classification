from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class MiscParams:
    """
    Data class representing miscellaneous parameters.

    Attributes:
    - data_path (str): The root directory for raw data.
    """
    data_path : str = './data.csv'
    random_state: int = 42
    target_col : str = 'ONET_NAME'
    text_col : str = 'TEXT'

@dataclass
class EmbedParams:
    sbert_model_name = 'all-MiniLM-L6-v2'
    # num_categories = 50 

@dataclass
class CatEmbedModelParams:
    epochs = 5
    learning_rate = 5e-3
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")