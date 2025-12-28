from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch

from training.featurizations import QPPNetFeaturization
from training.preprocessing.feature_statistics import FeatureType


class OperatorTree:

    def __init__(self, query_plan: dict, depth: int = 0):
        self.depth = depth
        self.planning_time: float = Optional[float]
        self.runtime: float = Optional[float]
        self.children: List[OperatorTree] = []
        self.encoded_features = None
        if isinstance(query_plan, SimpleNamespace):
            query_plan = vars(query_plan)
        current_children = query_plan.pop('Plans', [])
        self.properties: dict = query_plan
        for child in current_children:
            if isinstance(child, SimpleNamespace):
                child = vars(child)
            self.children.append(OperatorTree(child, depth + 1))
        self.node_type = self.properties.pop('Node Type')
        if self.node_type in ["Hash Join", "Nested Loop", "Merge Join"]:
            self.node_type = "Join"  # Joins have the same set of features according to https://arxiv.org/pdf/1902.00132

    def add_runtimes(self, runtime: float, planning_time: float):
        self.runtime = runtime
        self.planning_time = planning_time

    def has_cardinality(self) -> bool:
        return 'Actual Rows' in self.properties.keys()

    def get_cardinality(self) -> int:
        return self.properties['Actual Rows']

    def min_cardinality(self, actual_cardinality: int = np.inf) -> int:
        """Recursively look for minimal cardinality in plan """
        if self.has_cardinality():
            if self.get_cardinality() < actual_cardinality:
                actual_cardinality = self.get_cardinality()
        for child in self.children:
            actual_cardinality = child.min_cardinality(actual_cardinality)
        return actual_cardinality

    def encode_recursively(self, column_statistics: dict, feature_statistics: dict, featurization: QPPNetFeaturization):
        self.encoded_features = torch.Tensor(
            self.encode_features(self.properties, self.node_type, feature_statistics, column_statistics, featurization))
        for children in self.children:
            children.encode_recursively(column_statistics=column_statistics, feature_statistics=feature_statistics, featurization=featurization)

    @staticmethod
    def encode_features(properties: dict, node_type: str, feature_statistics: dict, column_statistics: dict, featurization: QPPNetFeaturization) -> List:
        """Encode features according to feature statistics"""
        features = []
        for feature_name in featurization.QPP_NET_OPERATOR_TYPES[node_type]:
            if "Scan" in node_type and feature_name in ["Min", "Max", "Mean"]:
                # Encode Min, Max or Mean according to column statistics.
                if "Filter" in properties.keys():
                    enc_value = OperatorTree.map_column_statistics(column_statistics=column_statistics,
                                                                   feature_name=feature_name,
                                                                   table_name=properties["Relation Name"],
                                                                   filter_condition=properties["Filter"])
                elif "Recheck Cond" in properties.keys():
                    enc_value = OperatorTree.map_column_statistics(column_statistics=column_statistics,
                                                                   feature_name=feature_name,
                                                                   table_name=properties["Relation Name"],
                                                                   filter_condition=properties["Recheck Cond"])
                else:
                    # Scan operators do not always have a Filter or a Recheck condition, so no column stats can
                    # be assigned. This is how it is done in QPPNet implementation, referred to as default value.
                    enc_value = 0

            else:
                if feature_name not in properties.keys():
                    # Hash Buckets sometimes are missing in the plan, so we use the default value
                    if feature_name == "Hash Buckets" and node_type == "Hash" or feature_name == "Peak Memory Usage" and node_type == "Hash":
                        value = feature_statistics[feature_name]['scale']
                        # print(f"Warning: {feature_name} not found in plan, using default value")

                    elif feature_name == "Parent Relationship" and node_type == "Join":
                        value = "Inner"
                        print(f"Warning: {feature_name} not found in plan, using default value")
                    elif feature_name == "Sort Method" and node_type == "Sort":
                        value = "quicksort"
                        print(f"Warning: {feature_name} not found in plan, using default value")
                    else:
                        raise ValueError(f"Feature {feature_name} not found for operator {node_type}")
                else:
                    value = properties[feature_name]

                if isinstance(value, list):
                    # In the TPC-H dataset, two sort keys can occur. As there is no specification in the paper,
                    # we just use the first sort key as a feature
                    if feature_name == "Sort Key":
                        value = [value[0]]
                    assert len(value) == 1, f"Found features {value} for feature {feature_name}"
                    value = value[0]

                if feature_statistics[feature_name].get('type') == str(FeatureType.numeric):
                    enc_value = feature_statistics[feature_name]['scaler'].transform(np.array([[value]])).item()

                # For categorical features, apply one hot encoding according to original implementation
                elif feature_statistics[feature_name].get('type') == str(FeatureType.categorical):
                    value_dict = feature_statistics[feature_name]['value_dict']
                    if value not in value_dict.keys():
                        raise ValueError(f"Value {value} not found in value dictionary of feature {feature_name}")
                    hot_idx = value_dict[value]
                    no_feats = feature_statistics[feature_name]['no_vals']
                    one_hot_vec = np.zeros(no_feats)
                    one_hot_vec[hot_idx] = 1
                    enc_value = one_hot_vec
                else:
                    raise NotImplementedError
            features.append(enc_value)

        # Convert to common numpy array
        features = np.concatenate([np.atleast_1d(elem) if not isinstance(elem, np.ndarray) else elem for elem in features])
        return features

    @staticmethod
    def map_column_statistics(column_statistics: dict, table_name: str, feature_name: str, filter_condition: str):
        for column_name in column_statistics[table_name].keys():
            if column_name in filter_condition:
                col_stats = column_statistics[table_name][column_name]
                if col_stats['datatype'] in ["int", "float"] and col_stats['num_unique'] > 1:
                    # Return min, mean and max scaled by max value as a simple scaling
                    value = column_statistics[table_name][column_name][feature_name.lower()]
                    max_val = column_statistics[table_name][column_name]["max"]
                    return value / max_val
                else:
                    return 0

        raise ValueError()

    def __str__(self):
        children_str = " ".join(str(child) for child in self.children)
        if children_str:
            return self.node_type + ", [" + children_str + "]"
        else:
            return self.node_type

    def __json__(self):
        json_dict = {
            'node_type': self.node_type,
            'properties': self.properties,
            'children': [child.to_json() for child in self.children]
        }
        if self.depth == 0:
            json_dict['runtime'] = self.planning_time
            json_dict['runtime'] = self.planning_time
        return json_dict

    def to_json(self):
        return self.__json__()


def operator_tree_from_json(query_plan: dict) -> OperatorTree:
    # operator_tree = OperatorTree(query_plan=vars(query_plan['Plan']))
    operator_tree = OperatorTree(query_plan=query_plan['Plan'])
    operator_tree.add_runtimes(runtime=query_plan['Execution Time']/1000, planning_time=query_plan['Planning Time']/1000)
    return operator_tree
