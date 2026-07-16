"""Resampling-based remediation techniques."""

from typing import Optional

import numpy as np
import pandas as pd


class ResamplingRemediation:
    """
    Resampling-based bias remediation.

    Provides methods for oversampling, undersampling, and hybrid approaches.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def oversample(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        target_ratio: float = 1.0,
        strategy: str = "random",
    ) -> pd.DataFrame:
        """
        Oversample minority groups.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        protected_attr : str
            Protected attribute column.
        target_ratio : float
            Target ratio relative to majority group (1.0 = equal size).
        strategy : str
            Oversampling strategy: 'random', 'smote'.

        Returns
        -------
        pd.DataFrame
            Resampled data.
        """
        if strategy == "smote":
            return self._smote_oversample(data, protected_attr, target_ratio)
        else:
            return self._random_oversample(data, protected_attr, target_ratio)

    def _random_oversample(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        target_ratio: float,
    ) -> pd.DataFrame:
        """Random oversampling with replacement."""
        np.random.seed(self.random_state)

        counts = data[protected_attr].value_counts()
        majority_group = counts.idxmax()
        majority_size = counts.max()
        target_size = int(majority_size * target_ratio)

        resampled_dfs = []

        for group in data[protected_attr].unique():
            group_df = data[data[protected_attr] == group]

            if group == majority_group:
                resampled_dfs.append(group_df)
            else:
                # Oversample with replacement
                if len(group_df) < target_size:
                    indices = np.random.choice(group_df.index, size=target_size, replace=True)
                    resampled_dfs.append(data.loc[indices])
                else:
                    resampled_dfs.append(group_df)

        result = pd.concat(resampled_dfs, ignore_index=True)
        return result.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

    def _smote_oversample(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        target_ratio: float,
    ) -> pd.DataFrame:
        """SMOTE oversampling (requires imbalanced-learn)."""
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ImportError(
                "imbalanced-learn required for SMOTE. Install with: pip install imbalanced-learn"
            ) from None

        # Separate features and protected attribute
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if protected_attr in numeric_cols:
            numeric_cols.remove(protected_attr)

        if not numeric_cols:
            raise ValueError("SMOTE requires numeric features")

        X = data[numeric_cols]
        y = data[protected_attr]

        # Calculate sampling strategy
        counts = y.value_counts()
        majority_size = counts.max()

        sampling_strategy = {
            group: max(int(majority_size * target_ratio), count) for group, count in counts.items()
        }

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=min(5, min(counts) - 1) if min(counts) > 1 else 1,
        )

        X_resampled, y_resampled = smote.fit_resample(X, y)

        result = pd.DataFrame(X_resampled, columns=numeric_cols)
        result[protected_attr] = y_resampled

        # Add back non-numeric columns (will have NaN for synthetic samples)
        non_numeric = [c for c in data.columns if c not in numeric_cols and c != protected_attr]
        for col in non_numeric:
            result[col] = np.nan

        return result

    def undersample(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        strategy: str = "random",
    ) -> pd.DataFrame:
        """
        Undersample majority groups.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        protected_attr : str
            Protected attribute column.
        strategy : str
            Undersampling strategy: 'random', 'tomek', 'enn'.

        Returns
        -------
        pd.DataFrame
            Resampled data.
        """
        np.random.seed(self.random_state)

        counts = data[protected_attr].value_counts()
        min_size = counts.min()

        resampled_dfs = []

        for group in data[protected_attr].unique():
            group_df = data[data[protected_attr] == group]

            if len(group_df) > min_size:
                indices = np.random.choice(group_df.index, size=min_size, replace=False)
                resampled_dfs.append(data.loc[indices])
            else:
                resampled_dfs.append(group_df)

        result = pd.concat(resampled_dfs, ignore_index=True)
        return result.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

    def stratified_split(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        test_size: float = 0.2,
        target_column: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified train/test split maintaining group proportions.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        protected_attr : str
            Protected attribute to stratify by.
        test_size : float
            Proportion for test set.
        target_column : str, optional
            Also stratify by target if provided.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Train and test DataFrames.
        """
        from sklearn.model_selection import train_test_split

        # Create stratification key
        if target_column:
            stratify_key = data[protected_attr].astype(str) + "_" + data[target_column].astype(str)
        else:
            stratify_key = data[protected_attr]

        train_df, test_df = train_test_split(
            data,
            test_size=test_size,
            stratify=stratify_key,
            random_state=self.random_state,
        )

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def hybrid_resample(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        target_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Hybrid approach: oversample minority and undersample majority.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        protected_attr : str
            Protected attribute column.
        target_size : int, optional
            Target size per group. If None, uses median group size.

        Returns
        -------
        pd.DataFrame
            Resampled data with all groups at target size.
        """
        np.random.seed(self.random_state)

        counts = data[protected_attr].value_counts()

        if target_size is None:
            target_size = int(counts.median())

        resampled_dfs = []

        for group in data[protected_attr].unique():
            group_df = data[data[protected_attr] == group]
            current_size = len(group_df)

            if current_size == target_size:
                resampled_dfs.append(group_df)
            elif current_size < target_size:
                # Oversample
                indices = np.random.choice(group_df.index, size=target_size, replace=True)
                resampled_dfs.append(data.loc[indices])
            else:
                # Undersample
                indices = np.random.choice(group_df.index, size=target_size, replace=False)
                resampled_dfs.append(data.loc[indices])

        result = pd.concat(resampled_dfs, ignore_index=True)
        return result.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
