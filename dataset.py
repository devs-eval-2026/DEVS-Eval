from datasets import load_dataset
import pandas as pd
import os

def import_dataset(quick_test=False):
    # Load the entire dataset
    if not quick_test:
        dataset = load_dataset("devs-eval-2026/DEVS-Eval", split="train")
    else:
        dataset = load_dataset("devs-eval-2026/DEVS-Eval", "quick_test", split="train")

    # convert the entire dataset to a Pandas DataFrame
    df = dataset.to_pandas()

    # Create directory if not exists:
    os.makedirs("../data/complete", exist_ok=True)

    # Save the dataset to a CSV file
    df.to_csv("../data/complete/data.csv", index=False)

if __name__ == "__main__":
    import_dataset()