import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error


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


# TODO modify input param dtypes based on Barab's encoding
# TODO Also return source as Mor's code?
def rand_for(
    params: dict,
    loss_func: str,
    feat_used: list,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> list[pd.DataFrame]:
    """Fits a random forest and evaluates it with chosen loss functions

    Args:
        params (dict): hyperparameters for the random forest
        loss_func (str): loss function
        feat_used (list): column names to be included in modelling
        X_train (pd.DataFrame):
        X_test (pd.DataFrame):
        y_train (pd.DataFrame):
        y_test (pd.DataFrame):

    Returns:
        list[pd.DataFrame]: rmse, rf
    """
    rf = RandomForestClassifier(**params).fit(X_train.loc[:, feat_used], y_train)
    test_preds = rf.predict(X_test.loc[:, feat_used])
    if loss_func == "rmse":
        rmse = mean_squared_error(y_true=y_test, y_pred=test_preds)

    return rmse, rf
