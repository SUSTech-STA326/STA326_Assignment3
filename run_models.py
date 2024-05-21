if __name__ == '__main__':
    import sys
    import torch

    print("Python version:")
    print(sys.version, "\n")

    print("PyTorch version:")
    print(torch.__version__)

    import pandas as pd
    import numpy as np
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from Builder import DataBuilder, Models, Process

    # Load the original training data
    df_train = pd.read_csv('./ml-1m/ml-1m.train.rating', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

    # Drop the 'timestamp' column
    df_train.drop(columns='timestamp', inplace=True)

    all_items = set(range(3706))

    # Prepare a dictionary to hold negative samples for each user
    negative_samples = {uid: [] for uid in df_train['user_id'].unique()}

    df_train['rating'] = 1

    # Generate negative samples for each user
    for user_id in negative_samples:
        # Get positive samples for this user
        pos_items = set(df_train[df_train['user_id'] == user_id]['item_id'].unique())

        # Calculate the potential negative items
        neg_items = list(all_items - pos_items)

        # Randomly select 8 negative items
        chosen_negatives = np.random.choice(neg_items, 8, replace=False)

        # Append to the negative samples dictionary
        negative_samples[user_id].extend(chosen_negatives)

    # Create a DataFrame for negative samples
    neg_data_list = []
    for user_id, items in negative_samples.items():
        for item in items:
            neg_data_list.append([user_id, item, 0])

    df_negatives = pd.DataFrame(neg_data_list, columns=['user_id', 'item_id', 'rating'])

    # Concatenate the negative samples to the original data
    df_final = pd.concat([df_train, df_negatives], ignore_index=True)
    df_final.to_csv('./ml-1m/ml-1m.train.rating.final', sep='\t', header=False, index=False)

    df_test = pd.read_csv('./ml-1m/ml-1m.test.rating', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df_test.drop(columns='timestamp', inplace=True)
    df_test['rating'] = 1
    df_test.to_csv('./ml-1m/ml-1m.test.rating.final', sep='\t', header=False, index=False)

    # Build Data
    ratings_dataset_train = DataBuilder.RatingData('./ml-1m/ml-1m.train.rating.final')
    ratings_dataset_test = DataBuilder.RatingData('./ml-1m/ml-1m.test.rating.final')

    # Create DataLoader instances for each dataset
    ratings_loader_train = DataLoader(ratings_dataset_train, batch_size=256, shuffle=True)
    ratings_loader_test = DataLoader(ratings_dataset_test, batch_size=256, shuffle=True)

    negative_dataset = DataBuilder.NegativeData('./ml-1m/ml-1m.test.negative')
    # print(len(negative_dataset))
    # print(negative_dataset.__getitem__(0))
    negative_loader = DataLoader(negative_dataset, batch_size=256, shuffle=False)

    num_users = ratings_dataset_train.ratings['user_id'].nunique()
    num_items = ratings_dataset_train.ratings['item_id'].nunique()
    max_item_id = ratings_dataset_train.ratings['item_id'].max()
    min_item_id = ratings_dataset_train.ratings['item_id'].min()

    num_factors = 8  # Number of latent factors for GMF
    mlp_layers = [64, 32, 16, 8]  # Layer configuration for MLP

    models = {
        'GMF': Models.GMF(num_users, num_items, num_factors),
        'MLP-0': Models.MLP_with_hidden_layers(num_users, num_items, num_factors, 0),
        'MLP-1': Models.MLP_with_hidden_layers(num_users, num_items, num_factors, 1),
        'MLP-2': Models.MLP_with_hidden_layers(num_users, num_items, num_factors, 2),
        'MLP-3': Models.MLP_with_hidden_layers(num_users, num_items, num_factors, 3),
        'MLP-4': Models.MLP_with_hidden_layers(num_users, num_items, num_factors, 4),
        'NeuMF': Models.NeuMF(num_users, num_items, num_factors, mlp_layers)
    }
    # print("GMF output shape example:", models['GMF'](torch.tensor([0]), torch.tensor([0])).shape)
    # print("MLP output shape example:", models['MLP-4'](torch.tensor([0]), torch.tensor([0])).shape)
    # print("NeuMF output shape example:", models['NeuMF'](torch.tensor([0]), torch.tensor([0])).shape)
    epochs = 100
    results = {name: {'train_loss': [], 'HR@10': [], 'NDCG@10': []} for name in models}

    # Training loop
    for name, model in models.items():
        print(f"Training {name}")
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            train_loss = Process.train(model, ratings_loader_train, optimizer, criterion)
            hr, ndcg = Process.evaluate(model, ratings_loader_test, negative_loader)

            # Store metrics
            results[name]['train_loss'].append(train_loss)
            results[name]['HR@10'].append(hr)
            results[name]['NDCG@10'].append(ndcg)

            # Optionally print the metrics
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}')

    # You can now use `results` dictionary to analyze or plot results
