from docarray import BaseDoc, DocList
from typing import Optional, Dict, Any, TypedDict, List, Annotated
# from caia.insight import QuickInsight, DeepInsight
import pandas as pd
from caia.representation import DatasetRepresentation
from caia.tools import Tool
from operator import add



class Dataset(BaseDoc):
    X_train: str
    X_test: str
    y_train: str
    y_test: str
    description: dict


class SemanticMemory(BaseDoc):
    dataset_old: Dataset
    # tools: Dict[str, Tool]
    model_object: Any
    model_code: str


class EpisodicMemory(BaseDoc):
    dataset_new: Dataset
    quick_insight: Dict[str, Any]
    deep_insight: Dict[str, Any]


class ModelScore(TypedDict):
    on_new_data: float
    on_old_data: float

class ImprovementEntry(TypedDict):
    # Original code that was improved
    previous_code: str
    # New code after improvement
    new_code: str
    # Which graph made the improvement (fast or slow)
    graph_type: str  # 'fast' or 'slow'
    # If slow graph, which strategy was used
    strategy_type: Optional[str]  # 'model_selection', 'hyperparameter_tuning', 'ensemble_method', or None for fast graph
    # Performance metrics
    metrics: Dict[str, ModelScore]  # Contains both old_model and new_model scores
    # Description of changes made
    changes_made: Dict[str, Any]
    # Whether the improvement was successful
    outcome: str  # 'success' or 'failure'
    # Overall accuracy changes
    improvements: Dict[str, float]  # Stores the improvements on both distributions

def create_improvement_entry(
    previous_code: str,
    new_code: str,
    graph_type: str,
    strategy_type: Optional[str],
    old_model_score: ModelScore,
    new_model_score: ModelScore,
    changes_made: Dict[str, Any]
) -> ImprovementEntry:
    """Helper function to create an ImprovementEntry with proper metrics calculation"""
    
    # Calculate improvements
    improvements = {
        'new_distribution': new_model_score['on_new_data'] - old_model_score['on_new_data'],
        'old_distribution': new_model_score['on_old_data'] - old_model_score['on_old_data']
    }
    
    # Determine outcome based on improvements
    outcome = 'success' if any(v > 0 for v in improvements.values()) else 'failure'
    
    return {
        'previous_code': previous_code,
        'new_code': new_code,
        'graph_type': graph_type,
        'strategy_type': strategy_type,
        'metrics': {
            'old_model': old_model_score,
            'new_model': new_model_score
        },
        'changes_made': changes_made,
        'outcome': outcome,
        'improvements': improvements
    }

class WorkingMemory(TypedDict):
    episodic_memory: DocList[EpisodicMemory]
    semantic_memory: Optional[SemanticMemory]

    threshold: float #5% higher than the reference model
    generations_fast_graph: Dict[str, Any]
    generations_slow_graph: Dict[str, Any]
    
    improvement_history: Annotated[List[ImprovementEntry], add]
