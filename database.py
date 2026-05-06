import pandas as pd
import os


def save_to_csv(
    input_type,
    user_query,
    caption,
    prediction
):

    file_name = "database.csv"

    new_data = {
        "input_type": input_type,
        "user_query": user_query,
        "caption": caption,
        "prediction": prediction
    }

    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df = pd.concat(
            [df, pd.DataFrame([new_data])],
            ignore_index=True
        )

    else:
        df = pd.DataFrame([new_data])

    df.to_csv(
        file_name,
        index=False
    )