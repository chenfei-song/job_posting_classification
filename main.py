import pandas as pd
import numpy as np

import time
import collections
import logging

from params import MiscParams, EmbedParams, CatEmbedModelParams
import utils, text_utils
from cat_embedding_model import train_model, CategoryEmbeddingModel
import evaluator, predictor

def main():
    # Configure the logger
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    # Create a logger instance
    logger = logging.getLogger(__name__)

    ############ data loading 
    logger.info("\n ================================== \n")

    # load data
    df = utils.read_csv_file(MiscParams.data_path)
    df = df.sample(frac=1).reset_index(drop=True).sample(n=1000)

    if df is not None:
        logger.info("CSV file successfully read!")
    else:
        logger.info("Failed to read CSV file!")
    
    # text data preparation
    df = text_utils.feature_eng(df)
    # ecode categories
    df, id_to_name_dict = text_utils.encode_categories(df)

    logger.info("Data pre-processed!")
    
    full_train_data, test_data  = utils.split_data(df)
    train_data, val_data = utils.split_data(full_train_data)

    full_train_df, full_train_categories = utils.separate_target(full_train_data)
    train_df, train_categories = utils.separate_target(train_data, target_col = 'ONET_ID')
    val_df, val_categories = utils.separate_target(val_data, target_col = 'ONET_ID')
    test_df, test_categories = utils.separate_target(test_data, target_col = 'ONET_ID')
    logger.info("Data splitted!")
    
    num_categories =  len(np.unique(df[MiscParams.target_col]))
    
    model = CategoryEmbeddingModel(EmbedParams.sbert_model_name, num_categories)

    train_texts, val_texts = train_df[MiscParams.text_col].values.tolist(), val_df[MiscParams.text_col].values.tolist()

    ############ Experimentation code 
    logger.info("\n ================================== \n")
    logger.info("Start training...")

    train_start_time = time.time()
    train_losses, val_losses, train_accs, val_accs = train_model(model, train_texts, train_categories, val_texts, val_categories, CatEmbedModelParams.epochs, CatEmbedModelParams.learning_rate, CatEmbedModelParams.batch_size, CatEmbedModelParams.device)
    train_end_time = time.time()
    logger.info("Model training finished!")
    logger.info(f"Training time: {train_end_time - train_start_time:.2f} seconds")
    # Visualize training & validation losses and accuracy 
    evaluator.gen_train_val_viz(train_losses, val_losses, train_accs, val_accs)
    
    ############ Experimentation code
    logger.info("\n ================================== \n")
    logger.info("Start predicting...") 
    pred_start_time = time.time()
    # retrieve embeddings for all categories
    category_embeddings = model.category_embedding.weight.detach().cpu().numpy()
    # get text embedding for test data
    # prediction
    pred_categories = predictor.predict(test_df, model = model, category_embeddings = category_embeddings, category_dict = id_to_name_dict, topn=5)
    pred_end_time = time.time()

    logger.info("Prediction finished!")
    logger.info(f"Prediction time: {pred_end_time - pred_start_time:.2f} seconds")
    pred_acc = evaluator.calculate_topn_accuracy(test_categories, pred_categories, n=5)
    
    ############ Experimentation code
    logger.info("\n ================================== \n")
    logger.info("Test case...") 
    print(f"\n ---- Job Title: ------ \n {list(test_df.TITLE)[-1]}")
    print(f"\n ---- ONET NAME: ------ \n {test_categories[-1]}")
    print(f"\n ---- Predicted ONET NAME: ------ \n {pred_categories[-1]}")

if __name__ == "__main__":
    main()
