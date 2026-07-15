"""Reweighting-based remediation techniques."""

from typing import Optional
import numpy as np
import pandas as pd


class ReweightingRemediation:
    """
    Sample reweighting for bias mitigation.
    
    Computes sample weights to balance group representation or outcomes
    without modifying the data itself.
    """
    
    def __init__(self):
        pass
    
    def inverse_frequency_weights(
        self,
        data: pd.DataFrame,
        protected_attr: str,
    ) -> pd.Series:
        """
        Compute inverse frequency weights.
        
        Gives higher weight to underrepresented groups.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        protected_attr : str
            Protected attribute column.
        
        Returns
        -------
        pd.Series
            Sample weights.
        """
        counts = data[protected_attr].value_counts()
        total = len(data)
        n_groups = len(counts)
        
        # Weight = total / (n_groups * count_for_group)
        weights = data[protected_attr].map(
            lambda x: total / (n_groups * counts[x])
        )
        
        return weights
    
    def balanced_weights(
        self,
        data: pd.DataFrame,
        protected_attr: str,
    ) -> pd.Series:
        """
        Compute weights to perfectly balance groups.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        protected_attr : str
            Protected attribute column.
        
        Returns
        -------
        pd.Series
            Sample weights that sum to 1 for each group.
        """
        counts = data[protected_attr].value_counts()
        n_groups = len(counts)
        
        # Each group should contribute equally (1/n_groups of total weight)
        target_weight_per_group = 1.0 / n_groups
        
        weights = data[protected_attr].map(
            lambda x: target_weight_per_group / counts[x]
        )
        
        # Normalize so weights sum to number of samples
        weights = weights * len(data)
        
        return weights
    
    def label_balancing_weights(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        target_column: str,
        positive_label: any = 1,
    ) -> pd.Series:
        """
        Compute weights to balance positive rates across groups.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        protected_attr : str
            Protected attribute column.
        target_column : str
            Target/label column.
        positive_label : any
            Value representing positive outcome.
        
        Returns
        -------
        pd.Series
            Sample weights for balanced positive rates.
        """
        # Calculate positive rate for each group
        positive_rates = data.groupby(protected_attr)[target_column].apply(
            lambda x: (x == positive_label).mean()
        )
        
        # Target: overall positive rate
        overall_positive_rate = (data[target_column] == positive_label).mean()
        
        weights = pd.Series(index=data.index, dtype=float)
        
        for group in data[protected_attr].unique():
            mask = data[protected_attr] == group
            group_rate = positive_rates[group]
            
            if group_rate == 0 or group_rate == 1:
                # Edge case: all positive or all negative
                weights[mask] = 1.0
            else:
                # Calculate weights for positive and negative samples
                positive_mask = mask & (data[target_column] == positive_label)
                negative_mask = mask & (data[target_column] != positive_label)
                
                # Adjust weights to achieve target positive rate
                pos_weight = overall_positive_rate / group_rate
                neg_weight = (1 - overall_positive_rate) / (1 - group_rate)
                
                weights[positive_mask] = pos_weight
                weights[negative_mask] = neg_weight
        
        # Normalize
        weights = weights / weights.mean()
        
        return weights
    
    def intersectional_weights(
        self,
        data: pd.DataFrame,
        protected_attrs: list[str],
    ) -> pd.Series:
        """
        Compute weights considering intersectionality.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        protected_attrs : list[str]
            List of protected attribute columns.
        
        Returns
        -------
        pd.Series
            Weights balancing intersectional groups.
        """
        # Create intersection key
        intersection = data[protected_attrs[0]].astype(str)
        for attr in protected_attrs[1:]:
            intersection = intersection + "_" + data[attr].astype(str)
        
        counts = intersection.value_counts()
        total = len(data)
        n_groups = len(counts)
        
        weights = intersection.map(
            lambda x: total / (n_groups * counts[x])
        )
        
        return weights
    
    def custom_target_weights(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        target_distribution: dict[str, float],
    ) -> pd.Series:
        """
        Compute weights to achieve a target distribution.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        protected_attr : str
            Protected attribute column.
        target_distribution : dict
            Target proportions {group: proportion}.
        
        Returns
        -------
        pd.Series
            Sample weights.
        """
        current_counts = data[protected_attr].value_counts()
        current_props = current_counts / len(data)
        
        # Compute weight multiplier for each group
        weight_multipliers = {}
        for group, target_prop in target_distribution.items():
            if group in current_props:
                current_prop = current_props[group]
                weight_multipliers[group] = target_prop / current_prop if current_prop > 0 else 0
            else:
                weight_multipliers[group] = 0
        
        weights = data[protected_attr].map(
            lambda x: weight_multipliers.get(x, 1.0)
        )
        
        # Normalize
        weights = weights / weights.mean()
        
        return weights
    
    def fairness_weights(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        target_column: str,
        fairness_metric: str = "demographic_parity",
    ) -> pd.Series:
        """
        Compute weights optimized for a fairness metric.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        protected_attr : str
            Protected attribute column.
        target_column : str
            Target column.
        fairness_metric : str
            'demographic_parity' or 'equalized_odds'.
        
        Returns
        -------
        pd.Series
            Fairness-optimized weights.
        """
        if fairness_metric == "demographic_parity":
            return self.label_balancing_weights(data, protected_attr, target_column)
        elif fairness_metric == "equalized_odds":
            # Simplified equalized odds reweighting
            return self._equalized_odds_weights(data, protected_attr, target_column)
        else:
            raise ValueError(f"Unknown fairness metric: {fairness_metric}")
    
    def _equalized_odds_weights(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        target_column: str,
    ) -> pd.Series:
        """Compute equalized odds reweighting."""
        weights = pd.Series(1.0, index=data.index)
        
        # For each (protected_group, true_label) combination
        for group in data[protected_attr].unique():
            for label in data[target_column].unique():
                mask = (data[protected_attr] == group) & (data[target_column] == label)
                
                # Count in this cell
                cell_count = mask.sum()
                
                # Count same label across all groups
                label_count = (data[target_column] == label).sum()
                
                # Count in this group
                group_count = (data[protected_attr] == group).sum()
                
                if cell_count > 0:
                    # Expected count under independence
                    expected = (label_count * group_count) / len(data)
                    
                    # Weight to achieve independence
                    weight = expected / cell_count
                    weights[mask] = weight
        
        # Normalize
        weights = weights / weights.mean()
        
        return weights
