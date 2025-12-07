import pandas as pd

# Load dataset and show a quick preview
if __name__ == "__main__":
    df = pd.read_csv("Titanic-Dataset.csv")
    print("Rows, columns:", df.shape)
    print(df.head())
    # Add your analysis code below
