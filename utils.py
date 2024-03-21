import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def read_csv_file(file_path):
    """
    Read a CSV file and return a Pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        # Perform any necessary data cleaning or preprocessing
        
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: Failed to read CSV file '{file_path}'.")
        print(f"Exception: {e}")
        return None

def split_data(data, split_ratio=0.2, group_col='ONET_NAME', min_samples=5, random_state=42):
    """
    Split data into train and test sets, ensuring stratified sampling within each group.

    Args:
        data (pd.DataFrame): The input DataFrame.
        split_ratio (float): The ratio of test set size to the total data size. Default is 0.2.
        group_col (str): The column name used for grouping the data. Default is 'ONET_NAME'.
        min_samples (int): The minimum number of samples required in a group for stratified sampling.
                           Groups with fewer samples will have their data entirely included in the training set.
                           Default is 5.
        random_state (int): Random seed for shuffling the data. Default is 42.

    Returns:
        pd.DataFrame, pd.DataFrame: Tuple containing train and test DataFrames.
    """
    # Split data into groups
    groups = data.groupby(group_col)

    # Initialize lists to store sampled data
    train_data = []
    test_data = []

    # Perform stratified sampling for each group
    for group_name, group_data in groups:
        # Handle groups with few samples
        if len(group_data) <= min_samples:
            train_data.append(group_data)
        else:
            # Split data within each group into train and test sets
            group_train, group_test = train_test_split(group_data, test_size=split_ratio, random_state=random_state)

            # Append sampled data to the lists
            train_data.append(group_train)
            test_data.append(group_test)

    # Concatenate sampled data from all groups
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)

    # Shuffle the data
    train_data = shuffle(train_data, random_state=random_state)
    test_data = shuffle(test_data, random_state=random_state)
    return train_data, test_data

def separate_target(data, target_col='ONET_NAME'):
    """
    Separate text data and target categories from the input DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing text data and target categories.
        target_col (str): The name of the target column in the DataFrame. Default is 'ONET_NAME'.

    Returns:
        pd.DataFrame, list: A tuple containing text data (DataFrame) and target categories (list).
    """
    # Split text and category
    text_data = data[['TITLE', 'BODY', 'TEXT']]
    categories = data[target_col].tolist()
    
    return text_data, categories




