from enum import Enum

class SupportedSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
