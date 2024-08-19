import numpy as np
import pandas as pd

def validate_data(data):
    try:
        np.array(data)
        return True
    except:
        return False

def preprocess_data(df):
    df.rename(columns={'Unnamed: 0': 'year'}, inplace=True)
    df.set_index('year', inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.round(0).astype(int)
    return df

def determine_target(df,target):
    x=df.drop(columns=[target])
    y=df[target]
    
    return x,y

def get_specific_df(df,section):
    return df.filter(regex=section)
    
def read_file(file):
    """
    Reads a file (CSV or XLSX) into a Pandas DataFrame.

    Parameters:
    - file: a file-like object (e.g., from Flask's request.files)

    Returns:
    - df: Pandas DataFrame containing the data
    """
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file, sep=';')
    elif file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        raise ValueError('Unsupported file format. Please upload a CSV or XLSX file.')
    
    return df