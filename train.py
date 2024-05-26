import os
import yaml
import torch
import wandb
import pandas as pd
import numpy as np
import argparse
from metrics import Metrics
from models import MLP, GMF, NeuMF
from data import Generator
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='training.log',
                    filemode='w')
logger = logging.getLogger()

class Trainer:
    def __init__(self, model, config, generator, evaluator):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.config = config
        self.generator = generator
        self.evaluator = evaluator
        self.evaluate_data = self.generator.get_evaluate_data
        self.optimizer = None
        self.crit = torch.nn.BCELoss()

    def get_optimizer(self):
        params = self.config
        if self.config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=params['lr'],
                                        momentum=params['momentum'],
                                        weight_decay=params['l2_regularization'])
        elif self.config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                        lr=params['adam_lr'],
                                        weight_decay=params['l2_regularization'])
        elif self.config['optimizer'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(),
                                            lr=params['lr'],
                                            alpha=params['alpha'],
                                            momentum=params['momentum'])
        self.optimizer = optimizer

    def _train_batch(self, users, items, ratings):
        if self.config['use_cuda']:
            users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
        self.optimizer.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def _train_epoch(self, train_loader, epoch_id):
        self.model.train()
        total_loss = 0
        for _, batch in enumerate(train_loader):
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self._train_batch(user, item, rating)
            total_loss += loss
        return total_loss
    
    def evaluate_epoch(self):
        self.model.eval()
        with torch.no_grad():
            test_users, test_items, negative_users, negative_items = \
                  self.evaluate_data[0], self.evaluate_data[1], self.evaluate_data[2], self.evaluate_data[3]
            if self.config['use_cuda']:
                test_users, test_items, negative_users, negative_items = \
                    test_users.to(self.device), test_items.to(self.device), negative_users.to(self.device), negative_items.to(self.device)                                                                                                                            
            test_scores = self.model(test_users, test_items).cpu()
            negative_scores = self.model(negative_users, negative_items).cpu()
            test_users, test_items, negative_users, negative_items = \
                test_users.cpu(), test_items.cpu(), negative_users.cpu(), negative_items.cpu()
            self.evaluator.subjects = [
                test_users.detach().view(-1).tolist(),
                test_items.detach().view(-1).tolist(),
                test_scores.detach().view(-1).tolist(),
                negative_users.detach().view(-1).tolist(),
                negative_items.detach().view(-1).tolist(),
                negative_scores.detach().view(-1).tolist()
            ]
        hit_ratio, ndcg = self.evaluator.cal_hit_ratio(), self.evaluator.cal_ndcg()
        return hit_ratio, ndcg
     
    def save(self, epoch_id, hit_ratio, ndcg):
        model_dir = self.config['model_dir'].format(self.config['alias'], epoch_id, hit_ratio, ndcg)
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        torch.save(self.model.state_dict(), model_dir)
    
    def train(self):
        wandb.init(project=self.config['alias'], config=self.config)
        self.get_optimizer()
        for epoch in range(self.config['num_epoch']):
            print('Epoch {} starts !'.format(epoch))
            print('-' * 80)
            train_loader = self.generator.get_train_loader(self.config['num_negative'], self.config['batch_size'])
            loss = self._train_epoch(train_loader, epoch_id=epoch)
            print('[Training Epoch {}], Loss {}'.format(epoch, loss))
            hit_ratio, ndcg = self.evaluate_epoch()
            logger.info(f"Epoch {epoch+1}/{self.config['num_epoch']}, Train Loss: {loss:.4f}, hit_ratio: {hit_ratio:.4f}, ndcg: {ndcg:.4f}")
            wandb.log({"Train Loss": loss, "Hit Ratio": hit_ratio, "NDCG" : ndcg})
            if epoch % 10 == 0: 
                self.save(epoch, hit_ratio, ndcg)

def load_data():
    ml1m_dir = 'ml-1m/ratings.dat'
    ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    item_id = ml1m_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
    return ml1m_rating


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    os.environ['WANDB_API_KEY'] = "70b6e5b4818d2ba5ac193e506146212435b80f4d"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    config = load_config(args.config)
    ratings = load_data()
    generator = Generator(ratings)
    evaluator = Metrics(top_k=10)
    logger.info('-'*100)
    logger.info('-'*100)
    logger.info(f"Now is the information of training: {config['alias']}")
    # mlp = MLP(config)
    if config['experiment_type'] == 'mlp':
        model = MLP(config)
    elif config['experiment_type'] == 'gmf':
        model = GMF(config)
    elif config['experiment_type'] == 'neumf':
        model = NeuMF(config)
    trainer = Trainer(model, config, generator, evaluator)
    trainer.train()
    
if __name__ == "__main__":
    main()