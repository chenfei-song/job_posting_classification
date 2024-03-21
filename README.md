# job_posting_classification

# Overview
## Objectives
Assist labor market data inference with job classification results based on occupational categories (O*NET).

## Dataset
50k sample data parsed job postings (ID, Post date, job title, body text) and their associated O*NET classifications (ONET Name, ONET)

## Solution
Build an advanced classification model that ingests job postings and predicts topN most likely occupational categories based on a pre-trained language model.

# Approach
## Workflow

![workflow](https://github.com/chenfei-song/job_posting_classification/assets/22181694/8300237a-54d1-44ed-a3e6-49c56a47536d)

## Key Assumptions 
- Categories in the sample set are the full category set
- Categories are correctly labeled
- Rare categories are important for our business case (hard-to-fill roles & diversity)
- Job postings are english based

## Fine-tuned category embedding
### Fine-tune
- Group average embedding doesn’t consider negative classes
### Contrastive Loss
- Poor margin and lacking robustness of cross-entropy loss (800+ classes)
- Contrastive loss focuses on hard negatives and inherently compare cosine similarities
![contrastive_loss1](https://github.com/chenfei-song/job_posting_classification/assets/22181694/5db51f36-27a5-416d-855d-8d441ad2a9ac)
![contrastive_loss2](https://github.com/chenfei-song/job_posting_classification/assets/22181694/f7f2ccac-d0ac-4778-98da-d46f557a0ec6)


## Metrics
- Performance-based
    - Overall accuracy (Top 1/3/5)
    - Accuracy by job category
- Engineering-based
    - Training time
    - Inference time

# Analysis
The following and result performance is based on provided 50k samples, which is split into train set(32k), validation set (8k) and test set (10k), run on Google Colab T4 GPU.

## Data Split with stratification
Among total 833 categories, 387 of them have less than 10 records; 263 less than 5, 114 with single record.

## Feature Representation
Title embedding has better accuracy than Body or Title+Body

## Category Representation
- Average title embeddings in the same category has better accuracy than direct embedding
- Fine-tuned category embeddings further improves accuracy, but requires 26X train time

## Final Approach
Use title feature to generate job posting embedding with pre-trained SBERT model 
Use Fine-tuned category embedding model to generate category embedding
Final prediction is based on cosine similarity and KNN of title embedding and category embeddings

## Code Architecture
![code_architecture](https://github.com/chenfei-song/job_posting_classification/assets/22181694/926d096f-a318-4352-8fd7-5caabdf7d71a)

# Next Steps
Depending on business priorities and resource optimization, potential next steps could include:

## Category Segmentation
- Deep-dive into mis-classified categories
- Evaluate rare category performance (considering AdeptID specializes in hard-to-fill roles and value diversities)
- For rare categories, experiment data augmentation (split body into pieces)

## Representation improvement
- Deep-dive into job body text
Job body may contain company description, responsibilities, qualifications, benefits apart from job description. Further NLP extraction could be helpful to improve job posting representations.
- Utilize existing entity representation spaces
If skills could be extracted from the job description, existing tools (1,000 latent skill features) could be added into feature space to test if it could improve model performance.

## Model experimentation
- Context window
While the chosen model (‘all-MiniLM-L6-v2’) supports text up to 256 tokens, sequence length is limited to 128 tokens while training. Truncating to the recommended size (i.e. 128 tokens) gives better results than extending it to 256. (reference: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/discussions/52)
- Other pre-trained models
Experiment other pre-trained language models (could also have different maximum sequence length, e.g., 'BAAI/bge-small-en-v1.5' has a sequence length of 512 tokens) and compare model performance.
- Multilingual situation
Experiment multilingual pre-trained model ('paraphrase-xlm-r-multilingual-v1') if necessary.

## Engineering refactorization
- Embed into existing infrastructure and modules
- Save category embedding into vector store for faster inference
- Schedule model retraining for new job categories or data shift

# Appedix
Occupation taxonomy link: [https://www.onetcenter.org/taxonomy/2019/data_coll.html](url)
report link: [https://docs.google.com/document/d/1O986NNo2-gM9TsnF-ZxULFFwcnTgtaj4pjOPNmcntFQ/edit?usp=sharing](url)
