from docarray import BaseDoc, DocList
from typing import Optional, Dict, Any, TypedDict, List, Annotated
# from caia.insight import QuickInsight, DeepInsight
import pandas as pd
from caia.representation import DatasetRepresentation
from caia.tools import Tool
from operator import add

# class NewDataset(BaseDoc):
#     X: pd.DataFrame
#     y: Optional[pd.DataFrame]
#     description: dict = None

# class QuickInsight(BaseDoc):
#     quick_insight: dict

# class DeepInsight(BaseDoc):
#     deep_insight: dict



class Dataset(BaseDoc):
    X_train: str
    X_test: str
    y_train: str
    y_test: str
    description: dict

# class ReferenceDataset(BaseDoc):
#     X_train: pd.DataFrame
#     X_test: pd.DataFrame
#     y_train: pd.Series
#     y_test: pd.Series
#     description: dict


class SemanticMemory(BaseDoc):
    reference_dataset: Dataset
    # tools: Dict[str, Tool]
    model_object: Any
    model_code: str


class EpisodicMemory(BaseDoc):
    new_dataset: Dataset
    quick_insight: Dict[str, Any]
    deep_insight: Dict[str, Any]


class ImprovementEntry(TypedDict):
    previous_code: str
    new_code: str
    changes_made: Dict[str, Any]
    outcome: str  # 'success' or 'failure'
    accuracy_change: float

class WorkingMemory(TypedDict):
    episodic_memory: DocList[EpisodicMemory]
    semantic_memory: Optional[SemanticMemory]

    threshold: float #5% higher than the reference model
    generations_fast_graph: Dict[str, Any]
    generations_slow_graph: Dict[str, Any]
    # dataset_representation: DatasetRepresentation
    
    # New fields for the fast graph
    monitoring_report: str
    improvement_history: Annotated[List[ImprovementEntry], add]
