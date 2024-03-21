import re
from sklearn.preprocessing import LabelEncoder
import time

# Text cleaning
def preprocess_text(text):
    # Clean out any HTML tags
    text = re.sub(r'<.*?>', " ", text)
    # Remove excess whitespace
    text = re.sub(' +', ' ', text).strip()
    # remove URL
    text = re.sub(r'http\S+', '', text)
    return text

def feature_eng(df):
    df['BODY'] = df['BODY'].apply(preprocess_text)
    df['TITLE'] = df.apply(lambda row: str(row['TITLE_RAW']) if isinstance(row['TITLE_RAW'], str) else row['BODY'], axis = 1)
    df['TEXT'] = df.apply(lambda row: "Job title is " + str(row['TITLE_RAW']) + ". " + "Job description is " + row['BODY'] if isinstance(row['TITLE_RAW'], str) else "Job description is " + row['BODY'], axis = 1)
    return df

def encode_categories(df):
  label_encoder = LabelEncoder()
  # Fit label encoder on the category names and transform them to category IDs
  df['ONET_ID'] = label_encoder.fit_transform(df['ONET_NAME'])
  id_to_name_dict = df.groupby('ONET_ID')['ONET_NAME'].first().to_dict()
  return df, id_to_name_dict

def gen_feature_embedding(df, model, col = 'TEXT'):
  # generate embedding for given column
  start_time = time.time()
  embeddings = model.sbert_model.encode(df[col].values.tolist())
  end_time = time.time()
  print(f"embedding generation time for {col}: {end_time - start_time:.2f} seconds")
  # return train_embeddings, test_embeddings
  return embeddings
