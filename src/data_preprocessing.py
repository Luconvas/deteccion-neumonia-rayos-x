import os
import pandas as pd
import dask.dataframe as dd

def load_data(path):
    df = dd.read_csv(path)
    return df

def preprocess_data(df):
    # Ejemplo de preprocesamiento
    df = df.dropna()
    return df

if __name__ == "__main__":
    raw_data_path = 'data/raw/data.csv'
    processed_data_path = 'data/processed/data.csv'
    
    df = load_data(raw_data_path)
    df = preprocess_data(df)
    df.to_csv(processed_data_path, single_file=True)
