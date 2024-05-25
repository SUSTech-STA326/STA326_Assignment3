import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import json
from multiprocessing import Process
import logging

def setup_logging(log_dir):
    logging.basicConfig(
        filename=log_dir,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def run_experiment(config):
    setup_logging(config['log_dir'])
    alias = config['alias']
    logging.info(f"Running experiment {alias}")
    
    if "gmf" in alias:
        engine = GMFEngine(config)
    elif "mlp" in alias:
        engine = MLPEngine(config)
        logging.info(f"Model structure:\n{engine.model}")
    elif "nmf" in alias:
        engine = NeuMFEngine(config)
        logging.info(f"Model structure:\n{engine.model}")
    
    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        loss=engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        logging.info(f"Epoch {epoch} - Loss: {loss:.2f} HR: {hit_ratio:.4f}, NDCG: {ndcg:.4f}")

def start_experiment(experiment):
    run_experiment(experiment)


# Load Data
ml1m_dir = 'data/ml-1m/ratings.dat'
ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
# Reindex
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
# DataLoader for training
sample_generator = SampleGenerator(ratings=ml1m_rating)
evaluate_data = sample_generator.evaluate_data

# 读取配置文件并运行实验
if __name__ == '__main__':
    with open('configs.json', 'r') as file:
        config = json.load(file)
    
    processes = []
    for experiment in config['experiments']:
        p = Process(target=start_experiment, args=(experiment,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()





