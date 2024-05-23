import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_X_y_split(df: pd.DataFrame, y_colname: str, test_ratio: float = 0.2):
    """Creates test/train and X/y split of a df

    Args:
        df (pd.DataFrame): your data frame
        y_colname (str): name of target variable
        test_ratio (float, optional): ratio of test set. Defaults to 0.2.

    Returns:
        X_train, X_test, y_train, y_test (pd.DataFrame): test/train split and X/y split dfs
    """
    retlist = train_test_split(
        df.drop(y_colname, axis=1),
        df[y_colname],
        test_size=test_ratio,
        random_state=72,
    )

    return [
        (
            pd.DataFrame(f, columns=[f for f in df.columns if not f == y_colname])
            if i < 2
            else pd.DataFrame(f, columns=[y_colname])
        )
        for i, f in enumerate(retlist)
    ]
