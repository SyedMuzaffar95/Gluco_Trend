import pandas as pd
def load_dataset():
    df = pd.read_csv('dataset/GlucoBench_benchmark_dataset.csv')
    return df


# --------------------------------------------------
# FIRST 5 ROWS
# --------------------------------------------------
def get_head(df):
    return df.head()
# --------------------------------------------------
# DATASET INFORMATION
# -------------------------------------------------
def get_info(df):
    return df.info()
# --------------------------------------------------
# STATISTICAL DESCRIPTION
# --------------------------------------------------

def get_description(df):

    return df.describe()


# --------------------------------------------------
# DATASET SHAPE
# --------------------------------------------------

def get_shape(df):

    return df.shape


# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

def get_missing_values(df):

    return df.isnull().sum()


# --------------------------------------------------
# LAST 5 ROWS
# --------------------------------------------------

def get_tail(df):

    return df.tail()


# --------------------------------------------------
# DUPLICATE ROWS
# --------------------------------------------------

def get_duplicates(df):

    return df.duplicated().sum()


# --------------------------------------------------
# UNIQUE VALUES
# --------------------------------------------------

def get_unique_values(df):

    return df.nunique()


# --------------------------------------------------
# COLUMN NAMES
# --------------------------------------------------

def get_columns(df):

    return df.columns.tolist()