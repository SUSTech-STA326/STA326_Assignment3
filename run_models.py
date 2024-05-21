import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from Builder import DataBuilder, Models, Process


if __name__ == '__main__':
    # 查看python 和 pytorch 的版本
    print("Python version:")
    print(sys.version)
    print()
    print("PyTorch version:")
    print(torch.__version__)

    # 数据预处理
    data_preprocess = DataBuilder.Preprocess("./ml-1m/ml-1m.train.rating",
                                             "./ml-1m/ml-1m.test.rating")
    data_preprocess.generate_final()

    # Build Data
    ratings_train = DataBuilder.RatingData('./ml-1m/ml-1m.train.rating.final')
    ratings_test = DataBuilder.RatingData('./ml-1m/ml-1m.test.rating.final')
    negative_dataset = DataBuilder.NegativeData('./ml-1m/ml-1m.test.negative')

    # Create DataLoader instances for each dataset
    ratings_loader_train = DataLoader(ratings_train, batch_size=256, shuffle=True)
    ratings_loader_test = DataLoader(ratings_test, batch_size=256, shuffle=True)
    negative_loader = DataLoader(negative_dataset, batch_size=256, shuffle=False)

    all_users = ratings_train.ratings['user_id'].nunique()
    all_items = ratings_train.ratings['item_id'].nunique()

    num_factors = 8  # Number of latent factors
    mlp_layers = [64, 32, 16, 8]  # Layer configuration for MLP

    models = {
        'GMF': Models.GMF(all_users, all_items, num_factors),
        'MLP-0': Models.MLP_with_hidden_layers(all_users, all_items, num_factors, 0),
        'MLP-1': Models.MLP_with_hidden_layers(all_users, all_items, num_factors, 1),
        'MLP-2': Models.MLP_with_hidden_layers(all_users, all_items, num_factors, 2),
        'MLP-3': Models.MLP_with_hidden_layers(all_users, all_items, num_factors, 3),
        'MLP-4': Models.MLP_with_hidden_layers(all_users, all_items, num_factors, 4),
        'NeuMF': Models.NeuMF(all_users, all_items, num_factors, mlp_layers)
    }

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

    Process.draw(results, path="./output.png")   # 可视化并保存
    # You can now use `results` dictionary to analyze or plot results
