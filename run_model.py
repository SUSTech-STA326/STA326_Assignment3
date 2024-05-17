import torch.nn as nn
from model import *
import numpy as np
import os
import random
from tqdm import tqdm
from dataset import *
import torch.optim as optim
from torch.utils.data import DataLoader
from evaluate import *


def model_train(config, num_of_negatives=4, preprocessed_filepath = "preprocessed_data/ml.pkl", batch_size=256, num_of_epochs=30, seed=2024):
    ####* set seed and device
    seed_everything(seed)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    ####* init dataset and dataloader
    rating_mat, num_of_user, num_of_item, negative_sample_list, testing_ratings_list = init_train_data(preprocessed_filepath)
    # print("after init",num_of_user, num_of_item)
    train_dataset = RatingDataset(rating_mat, negative_sample_list, num_of_user, num_of_item, num_of_negatives)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ####* set model, loss, and optimizer
    config['num_users'] = num_of_user
    config['num_items'] = num_of_item
    print("config of model: \n",config)
    model = RecommenderModel(config)
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_hr, best_ndcg, best_epoch = 0, 0, 0

    ####* train model
    for epoch in tqdm(range(num_of_epochs)):
        model.train()
        for user, item, label in train_dataloader:
            optimizer.zero_grad()
            user,item,label = user.to(device), item.to(device), label.float().to(device)
            output = model(user, item)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        #### evaluate
        with torch.no_grad():
            model.eval()
            hits, ndcgs = evaluate_model(model, testing_ratings_list, negative_sample_list, 10, num_thread=1,device = device)
            hr = np.mean(hits)
            ndcg = np.mean(ndcgs)

        # Update best HR and NDCG
        if hr > best_hr:
            best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
            # torch.save(model.state_dict(), f"bestmodel/best_model_{config['model_type']}(factor-{config['latent_dim']},X-{config['layers_num(X)']}).pth")
        # print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Hit Ratio: {hr:.4f}, NDCG: {ndcg:.4f}')

    print(f'Best HR: {best_hr}, Best NDCG: {best_ndcg} at Epoch {best_epoch}')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

def init_train_data(preprocessed_file_path, rating_file_path = "Data/ml-1m.train.rating",negative_file_path = "Data/ml-1m.test.negative", testing_file_path = "Data/ml-1m.test.rating"):
    '''
    read files and preprocess them.
    if exits preprocessed file, the path should be conveyed by parameter `preprocessed_file_path`
    if not exits, the rating/negative/testing file path will be filled automatically by default, then the function will preprocess them and save them to the preprocessed file path.
    '''
    if os.path.exists(preprocessed_file_path):
        print(f"Loading preprocessed data from {preprocessed_file_path}")
        with open(preprocessed_file_path, 'rb') as f:
            rating_mat, num_of_user, num_of_item, negative_sample_list, testing_ratings_list = pickle.load(f)
    else:
        print(f"Preprocessing data beginning...")
        rating_mat, num_of_user, num_of_item = load_rating_file_as_sparse(rating_file_path)
        negative_sample_list = load_negative_file(negative_file_path)
        testing_ratings_list = load_rating_file_as_list(testing_file_path)
        print(f"Preprocessing finished and saving to {preprocessed_file_path}.")
        with open(preprocessed_file_path, 'wb') as f:
            pickle.dump((rating_mat, num_of_user, num_of_item, negative_sample_list, testing_ratings_list), f)

    return rating_mat, num_of_user, num_of_item, negative_sample_list, testing_ratings_list

if __name__ == "__main__":
    model_config1 = {
        'latent_dim': 8,
        "layers_num(X)" : 0,
        # 'layers': [ 32, 16, 8],
        'model_type': 'GMF'     #　MLP, NeuMF
    }

    model_config2 = {
        'latent_dim': 8,
        "layers_num(X)" : 3,
        'layers': [16, 32, 16, 8],   # layers[0] is the concat of latent user vector & latent item vector
        'model_type': 'MLP'     #　MLP, NeuMF
    }

    model_config3 = {
        'latent_dim': 8,
        "layers_num(X)" : 3,
        'layers': [16, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
        'model_type': 'NeuMF'     #　MLP, NeuMF
    }

    model_train(model_config1)
    print()
    print("-"*30)
    model_train(model_config2)
    print()
    print("-"*30)
    model_train(model_config3)
