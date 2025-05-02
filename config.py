from pathlib import Path

PROJECT_PATH: Path = Path(__file__).parent.resolve()

INDEX_SPAM_PATH         : Path = PROJECT_PATH / r"data/indexes/index_spam"
INDEX_HAM_PATH          : Path = PROJECT_PATH / r"data/indexes/index_ham"
DATASETS_PATH           : Path = PROJECT_PATH / r"data/datasets"
METADATA_MODEL_PATH     : Path = PROJECT_PATH / r"data/models/model.pth"
TRANSFORMER_MODEL_PATH  : Path = PROJECT_PATH / r"data/models/transformer_model.h5"
TESTING_PATH            : Path = PROJECT_PATH / r"data/datasets/testing.json"