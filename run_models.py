"""
Note!! To run this code, the data.zip folder needed to be unzip first.

This code will train model GMF, MLP and neuMF, 
then save the Models to folder 'checkpoints' and results(HR and NDCG as csv) to folder 'my_data_test'.
The dataset used to train the models is Movielens dataset
"""

import pandas as pd
import numpy as np
from data import SampleGenerator
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine


# use the same epoch num
epoch = 50

# Change config of each model here
gmf_config = {'alias': 'gmf_factor8neg4-implict',
                  'num_epoch': epoch,
                  'batch_size': 1024,
                  # 'optimizer': 'sgd',
                  # 'sgd_lr': 1e-3,
                  # 'sgd_momentum': 0.9,
                  # 'optimizer': 'rmsprop',
                  # 'rmsprop_lr': 1e-3,
                  # 'rmsprop_alpha': 0.99,
                  # 'rmsprop_momentum': 0,
                  'optimizer': 'adam',
                  'adam_lr': 1e-3,
                  'num_users': 6040,
                  'num_items': 3706,
                  'latent_dim': 32,
                  'num_negative': 4,
                  'l2_regularization': 0,  # 0.01
                  'weight_init_gaussian': True,
                  'use_cuda': False,
                  'device_id': 0,
                  'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': epoch,
#               'batch_size': 256,  # 1024,
              'batch_size': 1024,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
#               'latent_dim': 8, # = layers[0]*2
              'latent_dim': 64, # = layers[0]*2
              'num_negative': 4,
#               'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
#               'layers': [64, 32, 16, 8],
#               'layers': [64, 48, 32, 16, 8],
#               'layers': [64, 32, 32, 16, 8],
              'layers': [128, 96, 64, 32, 16, 8],
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'weight_init_gaussian': True,
              'use_cuda': False,
              'device_id': 0,
              'pretrain': False,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'neumf_factor8neg4',
                'num_epoch': epoch,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 64, # = layers[0]*2
                'num_negative': 4,
#                 'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
#                 'layers': [64, 32, 16, 8],
#                 'layers': [64, 48, 32, 16, 8],
#                 'layers': [64, 32,32, 16, 8],
                'layers': [128, 96, 64, 32, 16, 8],
                'l2_regularization': 0.0000001,
                'weight_init_gaussian': True,
                'use_cuda': False,
                'device_id': 0,
                'pretrain': False,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

# Load Data
ml1m_dir = 'data/ml-1m/ratings.dat'
ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')

# Reindex users and items
user_id = ml1m_rating[['uid']].drop_duplicates().reset_index(drop=True)
user_id['userId'] = user_id.index
ml1m_rating = pd.merge(ml1m_rating, user_id, on='uid', how='left')

item_id = ml1m_rating[['mid']].drop_duplicates().reset_index(drop=True)
item_id['itemId'] = item_id.index
ml1m_rating = pd.merge(ml1m_rating, item_id, on='mid', how='left')

ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))

# DataLoader for training and evaluation
sample_generator = SampleGenerator(ratings=ml1m_rating)
evaluate_data = sample_generator.evaluate_data


"""
Run GMF, MLP and neuMF sequentially
"""

##############################################################################
################################### GMF ######################################
gmf_hr_data = {'Step': [], 'Value': []}

gmf_ndcg_data = {'Step': [], 'Value': []}

# Train and evaluate the models
# GMF Model
gmf_engine = GMFEngine(gmf_config)
for epoch in range(gmf_config['num_epoch']):
        print(f'GMF - Epoch {epoch} starts!')
        train_loader = sample_generator.instance_a_train_loader(gmf_config['num_negative'], gmf_config['batch_size'])
        gmf_engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = gmf_engine.evaluate(evaluate_data, epoch_id=epoch)
        gmf_engine.save(gmf_config['alias'], epoch, hit_ratio, ndcg)

        # Save HR and NDCG values to dataframes
        gmf_hr_data['Step'].append(epoch)
        gmf_hr_data['Value'].append(hit_ratio)
        gmf_ndcg_data['Step'].append(epoch)
        gmf_ndcg_data['Value'].append(ndcg)

# Save HR and NDCG values to CSV files
gmf_hr_df = pd.DataFrame(gmf_hr_data)
gmf_hr_df.to_csv('my_data_test/gmf_HR_32.csv', index=False)

gmf_ndcg_df = pd.DataFrame(gmf_ndcg_data)
gmf_ndcg_df.to_csv('my_data_test/gmf_NDCG_32.csv', index=False)

##############################################################################
################################### MLP ######################################
mlp_hr_data = {'Step': [], 'Value': []}

mlp_ndcg_data = {'Step': [], 'Value': []}

# Train and evaluate the models

# MLP Model
mlp_engine = MLPEngine(mlp_config)
for epoch in range(mlp_config['num_epoch']):
    print(f'MLP - Epoch {epoch} starts!')
    train_loader = sample_generator.instance_a_train_loader(mlp_config['num_negative'], mlp_config['batch_size'])
    mlp_engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = mlp_engine.evaluate(evaluate_data, epoch_id=epoch)
    mlp_engine.save(mlp_config['alias'], epoch, hit_ratio, ndcg)
    
    # Save HR and NDCG values to dataframes
    mlp_hr_data['Step'].append(epoch)
    mlp_hr_data['Value'].append(hit_ratio)
    mlp_ndcg_data['Step'].append(epoch)
    mlp_ndcg_data['Value'].append(ndcg)
    
mlp_hr_df = pd.DataFrame(mlp_hr_data)
mlp_hr_df.to_csv('./my_data_test/mlp_HR_6layer.csv', index=False)

mlp_ndcg_df = pd.DataFrame(mlp_ndcg_data)
mlp_ndcg_df.to_csv('./my_data_test/mlp_NDCG_6layer.csv', index=False)

##############################################################################
################################### neuMF ######################################

neumf_hr_data = {'Step': [], 'Value': []}

neumf_ndcg_data = {'Step': [], 'Value': []}


# Train and evaluate the models

# NeuMF Model
neumf_engine = NeuMFEngine(neumf_config)
for epoch in range(neumf_config['num_epoch']):
    print(f'NeuMF - Epoch {epoch} starts!')
    train_loader = sample_generator.instance_a_train_loader(neumf_config['num_negative'], neumf_config['batch_size'])
    neumf_engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = neumf_engine.evaluate(evaluate_data, epoch_id=epoch)
    neumf_engine.save(neumf_config['alias'], epoch, hit_ratio, ndcg)
    
    # Save HR and NDCG values to dataframes
    neumf_hr_data['Step'].append(epoch)
    neumf_hr_data['Value'].append(hit_ratio)
    neumf_ndcg_data['Step'].append(epoch)
    neumf_ndcg_data['Value'].append(ndcg)
    
neumf_hr_df = pd.DataFrame(neumf_hr_data)
neumf_hr_df.to_csv('./my_data_test/neumf_HR_6layer.csv', index=False)

neumf_ndcg_df = pd.DataFrame(neumf_ndcg_data)
neumf_ndcg_df.to_csv('./my_data_test/neumf_NDCG_6layer.csv', index=False)