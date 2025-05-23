# models/PFA/PFA_helper.py
import pandas as pd

def add_pfa_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds kc_cor / kc_incor using chronological order of the raw timestamp string.
    """
    df_sorted = (
        df.sort_values(["user_id", "skill_name", "timestamp"])  # string sort is OK
          .copy()
    )

    grp = df_sorted.groupby(["user_id", "skill_name"])

    df_sorted["kc_cor"] = grp["correct"].cumsum().shift(fill_value=0)
    df_sorted["kc_incor"] = grp["correct"].transform(
        lambda s: (1 - s).cumsum().shift(fill_value=0)
    )

    df_sorted["score"] = df_sorted["correct"].astype(int)
    return df_sorted