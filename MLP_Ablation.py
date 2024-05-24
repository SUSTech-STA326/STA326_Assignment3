import pandas as pd
import numpy as np
from data import SampleGenerator
from MLP import MLPEngine
import matplotlib.pyplot as plt


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

#用来做消融实验的layer组合  layers[0] is the concat of latent user vector & latent item vector
layers = [[16],[16, 8],[16, 16, 8],[16, 32, 16, 8],[16, 64, 32, 16, 8]]
engines = []

for l in layers:
    config = {
        'alias':'MLP_' + str(len(l)),
        'num_epoch': 50,
        'batch_size': 256,
        'optimizer': 'adam',
        'adam_lr': 1e-3,
        'num_users': 6040,
        'num_items': 3706,
        'latent_dim': 8,
        'num_negative': 4,
        'layers': l,
        'l2_regularization': 0.0000001,
        'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
    }
    

    mlp_engine = MLPEngine(config)
    engines.append(mlp_engine)
    print(f"Start training {config['alias']} !")
    
    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        mlp_engine.train_an_epoch(train_loader, epoch_id=epoch)
        mlp_engine.evaluate(evaluate_data, epoch_id=epoch)

#print各个engine的结果
for engine in engines:
    print(f"Model: {engine.config['alias']}")
    print(f"HR@10: {engine.hr_list[-1]}")
    print(f"NDCG@10: {engine.ndcg_list[-1]}")
    print(f"Training Loss: {engine.loss_list[-1]}")
    print()

#画图        
metric_map={'HR@10': 'hr_list','NDCG@10': 'ndcg_list','Training Loss': 'loss_list'}
colors=['#DD4D3D','#9C4985','#00848E','#7E7D00','#56423E']
for k, metric_list in metric_map.items():
    plt.figure(figsize=(10, 8))
    for idx, engine in enumerate(engines):
        plt.plot(range(engine.config['num_epoch']), getattr(engine,metric_list), label=engine.config['alias'],color=colors[idx])
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(k, fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(f"mlp_{metric_list}.png", dpi=300)


