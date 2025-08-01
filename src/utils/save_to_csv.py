import os
import csv
from dataclasses import asdict
from training import TrainingResults


def save_to_csv(csv_path, results: TrainingResults):
    headers = {
        "train": ['Epoch', 'Steps', 'Train Loss', 'Train Accuracy', 'Learning Rate'],
        "test": ['Epoch', 'Steps', 'Test Loss', 'Test Accuracy'],
        "lr_bs": ['Epoch', 'Steps', 'Learning Rate', 'Batch Size'],
        "norm": ['Epoch', 'Steps', 'Full Gradient Norm']
    }

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    results_dict = asdict(results)

    for field_name, data in results_dict.items():
        if not data:
            continue

        if field_name not in headers:
            raise ValueError(f"Unsupported field: {field_name}")

        file_name = os.path.join(csv_path, f"{field_name}.csv")
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers[field_name])
            writer.writerows(data)
