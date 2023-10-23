import pandas as pd

def split_dataframe(df, ratios=[0.6, 0.2, 0.2]):
    """
    Splits a DataFrame into multiple DataFrames based on given ratios.

    Parameters:
    - df: DataFrame to be split
    - ratios: List of ratios for splitting. Default is [0.6, 0.2, 0.2]

    Returns:
    - List of DataFrames
    """
    if sum(ratios) != 1.0:
        raise ValueError("Ratios must sum to 1.")

    # Calculate split indices
    first_split = int(ratios[0] * len(df))
    second_split = first_split + int(ratios[1] * len(df))

    # Split the dataframe
    df1 = df.iloc[:first_split]
    df2 = df.iloc[first_split:second_split]
    df3 = df.iloc[second_split:]

    return df1, df2, df3

if __name__ == "__main__":
    kaggle_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    column_names = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
        "Rings",
    ]
    df = pd.read_csv(kaggle_url, names=column_names)
    df1, df2, df3 = split_dataframe(df.sample(frac=1))

    df1.to_csv("data/abalone_train.csv", index=False)
    df2.to_csv("data/abalone_test.csv", index=False)
    df3.to_csv("data/abalone_predict.csv", index=False)