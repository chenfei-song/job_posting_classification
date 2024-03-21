
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from sentence_transformers import SentenceTransformer


class TextDataset(Dataset):
    def __init__(self, texts, categories):
        """
        Args:
            texts (list): List of text inputs.
            categories (list): List of category labels.
        """
        self.texts = texts
        self.categories = categories

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns a tuple of text input and corresponding category label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Tuple containing text input and category label.
        """
        if self.categories is not None:
            return self.texts[idx], torch.tensor(self.categories[idx])
        else:
            return self.texts[idx]
        
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, hard_negative=True):
        """
        Args:
            margin (float): Margin value for the contrastive loss.
            hard_negative (bool): Whether to use hard negative mining.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.hard_negative = hard_negative

    def forward(self, similarities, categories):
        """
        Computes the contrastive loss.

        Args:
            similarities (torch.Tensor): Tensor of cosine similarities.
            categories (torch.Tensor): Tensor of category labels.

        Returns:
            torch.Tensor: Contrastive loss.
        """
        batch_size = categories.size(0)

        # Create a binary matrix where entry (i, j) is 1 if categories[i] == j else 0
        mask = (torch.arange(similarities.size(-1)).to(similarities.device).unsqueeze(0) == categories.unsqueeze(-1)).float()

        # Get positive similarities
        positive_similarities = torch.sum(mask * similarities, dim=-1)

        if self.hard_negative:
            # Mask out the positive similarities with '-inf'
            negative_similarities = similarities.masked_fill_(mask == 1, float('-inf'))

            # Get the max negative similarity for each sample
            negative_similarities, _ = torch.max(negative_similarities, dim=-1)
        else:
            # Mask out the positive similarities with 0.0
            negative_similarities = similarities.masked_fill_(mask == 1, 0.0)

            # Use all negative similarities
            negative_similarities = torch.sum(negative_similarities, dim=-1) / (similarities.size(-1) - 1)

        # Compute the contrastive loss
        loss = torch.sum(torch.clamp(negative_similarities - positive_similarities + self.margin, min=0.0))

        return loss / batch_size

class CategoryEmbeddingModel(nn.Module):
    def __init__(self, sbert_model_name, num_category, freeze_sbert=True):
        """
        Args:
            sbert_model_name (str): Name of the SentenceTransformer model.
            num_category (int): Number of categories.
            freeze_sbert (bool): Whether to freeze the SBERT model parameters.
        """
        super(CategoryEmbeddingModel, self).__init__()

        self.sbert_model = SentenceTransformer(sbert_model_name)
        if freeze_sbert:
            for param in self.sbert_model.parameters():
                param.requires_grad = False

        # Initializing the embedding matrix as a trainable parameter
        embedding_dim = self.sbert_model._first_module().auto_model.config.hidden_size
        self.category_embedding = nn.Embedding(num_category, embedding_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.loss_fn = ContrastiveLoss()

    def forward(self, text_inputs, categories=None):
        """
        Forward pass of the model.

        Args:
            text_inputs (torch.Tensor): Tensor of text inputs.
            categories (torch.Tensor): Tensor of category labels.

        Returns:
            tuple: Tuple containing predicted categories and loss (if categories is not None).
        """
        # Generate embeddings for the input texts using the SBERT model
        text_embeddings = self.sbert_model.encode(text_inputs, convert_to_tensor=True)
        text_embeddings = self.dropout(text_embeddings)

        # Compute cosine similarity between text embeddings and person embeddings
        similarities = self.cos_sim(text_embeddings.unsqueeze(1), self.category_embedding.weight.unsqueeze(0))

        # Get the topN category ID with the highest similarity for each text input
        predicted_categories = torch.argmax(similarities, dim=-1)

        if categories is not None:
            # Compute contrastive loss
            loss = self.loss_fn(similarities, categories)
            return predicted_categories, loss
        else:
            return predicted_categories, None

def train_model(model, train_texts, train_categories, val_texts, val_categories, epochs, learning_rate, batch_size, device):
    """
    Trains the model.

    Args:
        model (nn.Module): Model to train.
        train_texts (list): List of training text inputs.
        train_categories (list): List of training category labels.
        val_texts (list): List of validation text inputs.
        val_categories (list): List of validation category labels.
        epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for optimization.
        batch_size (int): Batch size for training.
        device (str): Device to use for training (e.g., 'cpu', 'cuda').

    Returns:
        tuple: Tuple containing lists of training losses, validation losses, training accuracy, validation accuracy
    """
    # Move the model to the specified device
    model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader for training and validation dat
    train_dataset = TextDataset(train_texts, train_categories)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TextDataset(val_texts, val_categories)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize lists to store losses and accuracies for each epoch
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        # Switch the model to training mode
        model.train()

        total_loss, total_samples = 0, 0
        train_preds, train_true = [], []

        # Create batches
        for batch_texts, batch_categories in train_loader:
          # batch_texts = batch_texts.to(device)
          # batch_categories = batch_categories.to(device)

          # Forward pass
          predictions, loss = model(batch_texts, batch_categories)

          # Backward pass and optimization
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

          # Update the total loss and total_samples
          total_loss += loss.item()
          total_samples += len(batch_categories)

          # Store predictions and true labels for accuracy calculation
          train_preds.extend(predictions.detach().cpu().numpy())
          train_true.extend(batch_categories.detach().cpu().numpy())

        # Compute the average loss and accuracy over the epoch
        avg_loss = total_loss / total_samples
        train_losses.append(avg_loss)
        train_acc = accuracy_score(train_true, train_preds)
        train_accs.append(train_acc)

        # Switch the model to evaluation mode
        model.eval()

        with torch.no_grad():
            total_val_loss, total_val_samples = 0, 0
            val_preds, val_true = [], []

            for val_batch_texts, val_batch_categories in val_loader:
              # val_batch_texts = val_batch_texts.to(device)
              # val_batch_categories = val_batch_categories.to(device)

              predictions, val_loss = model(val_batch_texts, val_batch_categories)

              total_val_loss += val_loss.item()
              total_val_samples += val_batch_categories.shape[0]
              val_preds.extend(predictions.detach().cpu().numpy())
              val_true.extend(val_batch_categories.detach().cpu().numpy())

            avg_val_loss = total_val_loss / total_val_samples
            val_losses.append(avg_val_loss)
            val_acc = accuracy_score(val_true, val_preds)
            val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    return train_losses, val_losses, train_accs, val_accs

# def save_model(model, model_save_name, save_dir):
#     save_path = os.path.join(save_dir, f'{model_save_name}.pt')
#     torch.save(model.state_dict(), save_path)
#     print(f"Model saved to {save_path}")
            