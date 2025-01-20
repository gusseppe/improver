from docarray import BaseDoc, DocList
from typing import Any, Dict, List, Callable
from caia.tools import Tool
import inspect


class ColumnRepresentation(BaseDoc):
    name: str = None
    description: str = None
    cat_or_num: str = None
    possible_values: Any = None
    data_type: str = None

    def as_dict(self):
        return {
            'name': self.name,
            'description': self.description,
            'cat_or_num': self.cat_or_num,
            'possible_values': self.possible_values,
            'data_type': self.data_type,
        }

class ToolRepresentation(BaseDoc):
    name: str = None
    description: str = None
    result: Any = None

    def as_dict(self):
        return {
            'name': self.name,
            'description': self.description,
            'result': self.result
        }

class DatasetRepresentation(BaseDoc):
    name: str = None
    description: str = None
    features: DocList[ColumnRepresentation] = None
    label: ColumnRepresentation = None

    def as_dict(self):
        return {
            'name': self.name,
            'description': self.description,
            'features': [feature.as_dict() for feature in self.features],
            'label': self.label.as_dict()
        }

def get_dataset_representation(semantic_memory) -> DatasetRepresentation:
    features_representations = []
    dataset_description = semantic_memory.reference_dataset.description

    for feature_name in dataset_description['FEATURES']:
        type_feature = 'categorical' if feature_name in dataset_description['CATEGORICAL_FEATURES'] else 'numerical'

        feature = ColumnRepresentation(
            name=feature_name, 
            description=dataset_description['FEATURE_DESCRIPTIONS'][feature_name],
            cat_or_num=type_feature,
            possible_values=dataset_description['COLUMN_VALUES'][feature_name],
            data_type=dataset_description['COLUMN_TYPES'][feature_name],
            # tools_results=DocList[ToolRepresentation](tools_results_)
        )
        features_representations.append(feature)

    label = ColumnRepresentation(
        name=dataset_description['LABEL'],
        description=dataset_description['LABEL_DESCRIPTION'],
        cat_or_num='categorical',
        possible_values=dataset_description['COLUMN_VALUES'][dataset_description['LABEL']],
        data_type=dataset_description['COLUMN_TYPES'][dataset_description['LABEL']]
    )

    dataset_representation = DatasetRepresentation(
        name=dataset_description['DATASET_TITLE'],
        description=dataset_description['DATASET_DESCRIPTION'],
        features=DocList[ColumnRepresentation](features_representations),
        label=label
    )
    
    return dataset_representation

