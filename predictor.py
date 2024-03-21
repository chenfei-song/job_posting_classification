from sklearn.neighbors import NearestNeighbors
import text_utils

def predict(df, model, category_embeddings, category_dict, col = 'TITLE', topn=5):
    """
    Predict the top 'topn' closest categories for each sample text in the DataFrame 'df' based on cosine similarity
    to the category embeddings.

    Args:
        df (pandas.DataFrame): DataFrame containing the sample texts.
        model: Model used to generate embeddings for the sample texts.
        category_embeddings (numpy.ndarray): Embeddings of the categories.
        category_dict (dict): Dictionary mapping category indices to category names.
        col (str): Name of the column in the DataFrame containing the sample texts.
        topn (int): Number of nearest neighbors to find.

    Returns:
        list of lists: A list where each element is a list of the 'topn' closest categories for each sample text.
    """
    # Generate embeddings for the input dataframe's column
    text_embeddings = text_utils.gen_feature_embedding(df, model = model, col = col)
    
    # Initialize a NearestNeighbors model using cosine similarity
    nn = NearestNeighbors(n_neighbors=topn, metric='cosine', algorithm='brute')
    nn.fit(category_embeddings)

    # Find the `topn` closest averaged category embeddings for each sample in embeddings
    _, indices = nn.kneighbors(text_embeddings)

    # Look up the categories corresponding to the indices of the closest averaged category embeddings
    closest_categories = [[category_dict[idx] for idx in sample_indices] for sample_indices in indices]

    return closest_categories