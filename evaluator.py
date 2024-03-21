import matplotlib.pyplot as plt
import pandas as pd

def gen_train_val_viz(train_losses, val_losses, train_accs, val_accs):
    # Plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Losses over time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot the training and validation accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Training accuracy')
    plt.plot(val_accs, label='Validation accuracy')
    plt.title('Accuracies over time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def calculate_topn_accuracy(true_categories, topn_pred_categories, n):
    """
    Calculate top-N prediction accuracy.

    Args:
        true_categories (list): List of true categories.
        topn_pred_categories (list of lists): List of lists containing predicted categories for the top-N predictions.
        n (int): Value of N for top-N predictions (e.g., 1, 3, 5).

    Returns:
        list: Top-N prediction accuracy for range 1 to N.
    """
    total_samples = len(true_categories)
    acc_list = []
    for i in range(1, n+1):
        correct_predictions = 0
        for true_category, pred_categories in zip(true_categories, topn_pred_categories):
            if true_category in pred_categories[:i]:
                correct_predictions += 1
        acc_list.append(correct_predictions / total_samples) 

    # Plot N vs the prediction accuracy
    plt.figure(figsize=(6, 3))
    
    plt.plot(list(range(1, n+1)), acc_list, label='Prediction accuracy')
    plt.title('N vs Prediction accuracy')
    plt.xlabel('topN')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    return acc_list


def calculate_mrr(true_categories, topn_pred_categories):
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        true_categories (list): List of true categories.
        topn_pred_categories (list of lists): List of lists containing predicted categories for the top-N predictions.

    Returns:
        float: Mean Reciprocal Rank (MRR).
    """
    reciprocal_ranks = []

    for true_category, pred_categories in zip(true_categories, topn_pred_categories):
        try:
            # Find the index of the true category in the predicted categories
            rank = pred_categories.index(true_category) + 1
            # Calculate reciprocal rank and append to the list
            reciprocal_ranks.append(1 / rank)
        except ValueError:
            # If true category not in predicted categories, assign a reciprocal rank of 0
            reciprocal_ranks.append(0)

    # Calculate the mean of reciprocal ranks
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr


def calculate_category_accuracy(true_categories, topn_pred_categories):
    """
    Calculate accuracy in each category.

    Args:
        true_categories (list): List of true categories.
        topn_pred_categories (list): List of topn predicted categories.

    Returns:
        pandas.DataFrame: DataFrame with category name, count, and accuracy.
    """
    # Combine true and predicted categories into a DataFrame
    df = pd.DataFrame({'True_Category': true_categories, 'Pred_Category': [topn_pred_category[0] for topn_pred_category in topn_pred_categories]})

    # Group by true category
    grouped = df.groupby('True_Category')

    # Initialize lists to store category names, counts, and accuracies
    category_names = []
    category_counts = []
    category_accuracies = []

    for category, group in grouped:
        # Count occurrences of each category
        category_count = len(group)
        category_names.append(category)
        category_counts.append(category_count)

        # Calculate accuracy
        accuracy = round(sum(group['True_Category'] == group['Pred_Category']) / category_count, 4)
        category_accuracies.append(accuracy)

    # Create DataFrame
    result_df = pd.DataFrame({'Category_Name': category_names, 'Category_Count': category_counts, 'Category_Accuracy': category_accuracies})
    result_df = result_df.sort_values(by='Category_Accuracy', ascending=False)

    return result_df

