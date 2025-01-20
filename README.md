# Protein Embedding Space Alignment 🧬

A PyTorch model that learns to optimize network embedding weights to better align protein embeddings with their functional relationships.

## Overview 🔍

This project implements a simple but effective model that learns to weight dimensions in protein network embeddings. By training on pairs of proteins that share (or don't share) GO terms, the model discovers which embedding dimensions are most important for capturing functional relationships in the STRING protein network space.

## Features ⭐

- Processes protein network embeddings from STRING database (v12.0)
- Uses GO term annotations as a supervision signal
- Implements a lightweight weighted cosine similarity model
- Learns dimension-specific importance weights
- Includes visualization tools for weight analysis

## Requirements 📋

torch
numpy
pandas
scikit-learn
h5py
matplotlib
seaborn

## Model Architecture 🏗️

The core model is elegantly simple - it learns a single vector of weights to reweight embedding dimensions:

class EmbeddingWeights(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(embedding_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, emb1, emb2):
        # Apply weights to embeddings
        weighted_emb1 = emb1 * self.weights
        weighted_emb2 = emb2 * self.weights

        # Compute cosine similarity and apply sigmoid
        cos_sim = nn.functional.cosine_similarity(weighted_emb1, weighted_emb2, dim=1)
        return self.sigmoid(cos_sim)

This architecture:
- Learns one weight per embedding dimension
- Applies same weights to both proteins in a pair
- Uses cosine similarity to measure alignment
- Outputs a similarity score between 0 and 1

## Training Data Preparation 📊

The model is trained on protein pairs labeled based on shared GO terms:
- Positive pairs: Proteins that share at least one GO term
- Negative pairs: Proteins that share no GO terms

Common GO terms (appearing in >0.5% of proteins) are filtered out to focus on specific functional relationships.

## Training 🚀

Training uses:
- Binary Cross Entropy Loss
- Adam optimizer
- Balanced batches of positive and negative pairs
- Early stopping based on validation F1 score

## Results Analysis 📈

The model learns which dimensions in the STRING network embeddings are most important for capturing functional relationships. Analysis shows:
- Distribution of learned weights
- Most/least important embedding dimensions
- Performance metrics (F1 score, accuracy)

## Usage 💻

1. Load and prepare data:
train_pairs_data, embeddings = prepare_data(train_pairs, train_nonpairs, network_vector_df)

2. Create dataset and dataloader:
train_dataset = PairDataset(train_pairs_data, embeddings)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

3. Train model:
model = EmbeddingWeights()
trained_model = train_model(model, train_loader, val_loader)

4. Analyze weights:
learned_weights = model.weights.detach().numpy()

