"""
This function produces the performance results for slices of the data
"""
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from ml.utils import inference, compute_model_metrics
from ml.data import process_data


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def get_sliced_performance_metrics(data_path, model_path, encoder_path, lb_path):
    # Load data into pd dataframe
    data = pd.read_csv(data_path)

    # Load model, encoder and lb
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)

    # Get test set: Use the same random_state as the training
    _, test = train_test_split(data, random_state=42, test_size=0.2)

    # Slice data and get performance metrics for each slice
    for feature in cat_features:
        for entry in test[feature].unique():
            temp_df = test[test[feature] == entry]
            X_test, y_test, _, _ = process_data(
                temp_df, cat_features, label="salary", training=False,
                encoder=encoder, lb=lb
            )
            y_pred = inference(model, X_test)
            precision, recall, f_beta = compute_model_metrics(y_test, y_pred)
            print(f"{feature}: {entry}; Precision: {precision}, Recall: {recall}, Fbeta: {f_beta}\n")
            with open("../../sliced_data_performance/slice_output.txt", 'a') as file:
                file.write(f"{feature} = {entry}; Precision: {precision}, Recall: {recall}, Fbeta: {f_beta}\n")


if __name__ == "__main__":
    data_path = "../../data/census.csv"
    model_path = "../../model/model.pkl"
    encoder_path = "../../model/encoder.pkl"
    lb_path = "../../model/lb.pkl"
    get_sliced_performance_metrics(data_path, model_path, encoder_path, lb_path)
