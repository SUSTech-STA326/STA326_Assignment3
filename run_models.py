import pandas as pd
import numpy as np
from GMF import GMFEngine
from MLP import MLPEngine
from NeuMF import NeuMFEngine
from data import SampleGenerator
import matplotlib.pyplot as plt

gmf_config = {'alias': 'gmf_factor8neg4-implict',
              'num_epoch': 50,
              'batch_size': 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0,  # 0.01
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': 50,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
             }

neumf_config = {'alias': 'neumf_factor8neg4',
                'num_epoch': 50,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16, 64, 32, 16, 8], 
                'l2_regularization': 0.0000001,
                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

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

gmf_engine= GMFEngine(gmf_config)
mlp_engine= MLPEngine(mlp_config)
neu_engine= NeuMFEngine(neumf_config)

# train the GMF, MLP, and NeuMF models on the MovieLens dataset and output the evaluation metrics HR@10 and NDCG@10 for each model
for epoch in range(gmf_config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(gmf_config['num_negative'], gmf_config['batch_size'])
    gmf_engine.train_an_epoch(train_loader, epoch_id=epoch)
    gmf_engine.evaluate(evaluate_data, epoch_id=epoch)

for epoch in range(mlp_config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(mlp_config['num_negative'], mlp_config['batch_size'])
    mlp_engine.train_an_epoch(train_loader, epoch_id=epoch)
    mlp_engine.evaluate(evaluate_data, epoch_id=epoch)

for epoch in range(neumf_config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(neumf_config['num_negative'], neumf_config['batch_size'])
    neu_engine.train_an_epoch(train_loader, epoch_id=epoch)
    neu_engine.evaluate(evaluate_data, epoch_id=epoch)


metric_map={'HR@10': 'hr_list','NDCG@10': 'ndcg_list','Training Loss': 'loss_list'}
#三个指标绘图
for k, metric_list in metric_map.items():
    plt.figure(figsize=(10, 8))
    plt.plot(range(gmf_config['num_epoch']), getattr(gmf_engine,metric_list), color='#77C80B', label='GMF')
    plt.plot(range(mlp_config['num_epoch']), getattr(mlp_engine,metric_list), color='#00ABFF',  label='MLP')
    plt.plot(range(neumf_config['num_epoch']), getattr(neu_engine,metric_list), color='#D1117C', label='NeuMF')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(k, fontsize=14)
    plt.title(k+' for MovieLens', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(metric_list+'.png', dpi=300)

#打印模型最终结果
print("After training", neumf_config['num_epoch'], "epochs:")
print("HR@10 GMF:", gmf_engine.hr_list[-1])
print("HR@10 MLP:", mlp_engine.hr_list[-1])
print("HR@10 NeuMF:", neu_engine.hr_list[-1])
print("NDCG@10 GMF:", gmf_engine.ndcg_list[-1])
print("NDCG@10 MLP:", mlp_engine.ndcg_list[-1])
print("NDCG@10 NeuMF:", neu_engine.ndcg_list[-1])