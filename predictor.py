from sklearn.neighbors import NearestNeighbors
import text_utils

def predict(test_df, model, category_embeddings, category_dict, col = 'TITLE', topn=5):
    """
    For each sample in `embeddings`, find the `topn` closest categories based on the cosine similarity
    to the `category_embeddings`.

    Args:
        text_embeddings (numpy.ndarray): Embeddings of the sample texts for which to find closest categories.
        category_embeddings (numpy.ndarray): The embeddings of the categories.
        categories (list): The categories corresponding to the `category_embeddings`.
        topn (int): The number of nearest neighbors to find.

    Returns:
        list of lists: A list where each element is a list of the `topn` closest categories for each sample.
    """
    text_embeddings = text_utils.gen_feature_embedding(test_df, model = model, col = col)
    
    # Initialize a NearestNeighbors model using cosine similarity
    nn = NearestNeighbors(n_neighbors=topn, metric='cosine', algorithm='brute')
    nn.fit(category_embeddings)

    # Find the `topn` closest averaged category embeddings for each sample in embeddings
    _, indices = nn.kneighbors(text_embeddings)

    # Look up the categories corresponding to the indices of the closest averaged category embeddings
    closest_categories = [[category_dict[idx] for idx in sample_indices] for sample_indices in indices]

    return closest_categories