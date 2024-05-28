import argparse
import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator

parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--method', type=str, choices=['gmf','mlp','neumf'],required=True, help='Path to the dataset JSON file')
parser.add_argument('--factor', type=int, choices=[8], required=True, help='factor')
parser.add_argument('--neg', type=int, choices=[0,1,2,3,4], required=True, help='neg')
args = parser.parse_args()

gmf_config = {'alias': f"gmf_factor{args.factor}neg{args.neg}-implict",
              'num_epoch': 200,
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
              'latent_dim': args.factor ,
              'num_negative': 4 ,
              'l2_regularization': 0,  # 0.01
              'weight_init_gaussian': True,
              'use_cuda': True,
              'device_id': 0,
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': f"mlp_factor{args.factor}neg{args.neg}_bz256_166432168_pretrain_reg_0.0000001",
              'num_epoch': 200,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': args.factor ,
              'num_negative': args.neg ,
              'layers': [16, 64, 32],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'weight_init_gaussian': True,
              'use_cuda': True,
              'device_id': 0,
              'pretrain': False,
              'pretrain_mf': 'checkpoints/{}'.format(f"gmf_factor{args.factor}neg{args.neg}_Epoch100_HR0.6391_NDCG0.2852.model"),
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': f"neumf_factor{args.factor}neg{args.neg}",
                'num_epoch': 200,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': args.factor,
                'latent_dim_mlp': args.factor,
                'num_negative': 4,
                'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.0000001,
                'weight_init_gaussian': True,
                'use_cuda': True,
                'device_id': 0,
                'pretrain': False,
                'pretrain_mf': 'checkpoints/{}'.format(f"gmf_factor{args.factor}neg{args.neg}_Epoch100_HR0.6391_NDCG0.2852.model"),
                'pretrain_mlp': 'checkpoints/{}'.format(f"mlp_factor{args.factor}neg{args.neg}_Epoch100_HR0.5606_NDCG0.2463.model"),
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
# Specify the exact model
if args.method=='gmf':
    config = gmf_config
    engine = GMFEngine(config)
elif args.method=='mlp':
    config = mlp_config
    engine = MLPEngine(config)
elif args.method== 'neumf':
    config = neumf_config
    engine = NeuMFEngine(config)
for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    print(f"HR:{hit_ratio}, NDCG:{ndcg}")
    if epoch%10==0:
        engine.save(config['alias'], epoch, hit_ratio, ndcg)
