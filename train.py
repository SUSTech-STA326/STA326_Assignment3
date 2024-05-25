import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import threading


import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(script_dir)

gmf_config = {'alias': 'gmf',
              'num_epoch': 50,
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
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0,  # 0.01
              'weight_init_gaussian': True,
              'use_cuda': True,
              'device_id': 0,
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp',
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
              'weight_init_gaussian': True,
              'use_cuda': True,
              'device_id': 0,
              'pretrain': False,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'neumf',
                'num_epoch': 50,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.0000001,
                'weight_init_gaussian': True,
                'use_cuda': True,
                'device_id': 0,
                'pretrain': False,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

result=pd.DataFrame(columns=['alias','epoch','hit_ratio','ndcg'])

def train_model(engine, config, model_name,result):
    sample_generator = SampleGenerator(ratings=ml1m_rating)
    evaluate_data = sample_generator.evaluate_data
    print(f"Training {model_name} starts!")
    
    for epoch in range(config['num_epoch']):
        print(f'Epoch {epoch} for {model_name} starts!')
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        engine.save(config['alias'], epoch, hit_ratio, ndcg)
        print(config['alias'], epoch, hit_ratio, ndcg)
        result=pd.concat([result, pd.DataFrame({'alias': [config['alias']], 
                                                'epoch': [epoch],
                                                'hit_ratio': [hit_ratio],
                                                'ndcg': [ndcg]})], ignore_index=True)

    print(f"Training {model_name} finished!")
    return result


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
# sample_generator = SampleGenerator(ratings=ml1m_rating)
# evaluate_data = sample_generator.evaluate_data
# # Specify the exact model
# # config = gmf_config
# # engine = GMFEngine(config)
# # config = mlp_config
# # engine = MLPEngine(config)
# config = neumf_config
# engine = NeuMFEngine(config)
# for epoch in range(config['num_epoch']):
#     print('Epoch {} starts !'.format(epoch))
#     print('-' * 80)
#     train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
#     engine.train_an_epoch(train_loader, epoch_id=epoch)
#     hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
#     engine.save(config['alias'], epoch, hit_ratio, ndcg)
gmf_engine = GMFEngine(gmf_config)
result=train_model(gmf_engine, gmf_config, "GMF",result.copy())

mlp_engine = MLPEngine(mlp_config)
result=train_model(mlp_engine, mlp_config, "MLP",result.copy())

neumf_engine = NeuMFEngine(neumf_config)
result=train_model(neumf_engine, neumf_config, "NeuMF",result.copy())

print('Done')

# 创建模型引擎

# gmf_engine = GMFEngine(gmf_config)
# mlp_engine = MLPEngine(mlp_config)
# neumf_engine = NeuMFEngine(neumf_config)

# # 创建线程
# thread_gmf = threading.Thread(target=train_model, args=(gmf_engine, gmf_config, "GMF",result))
# thread_mlp = threading.Thread(target=train_model, args=(mlp_engine, mlp_config, "MLP",result))
# thread_neumf = threading.Thread(target=train_model, args=(neumf_engine, neumf_config, "NeuMF",result))

# # 启动线程
# thread_gmf.start()
# thread_mlp.start()
# thread_neumf.start()

# # 等待线程完成
# thread_gmf.join()
# thread_mlp.join()
# thread_neumf.join()


result.to_csv('./result.csv')

print('gmf metric:')
print(result[result['alias']=='gmf'].tail(1))

print('mlp metric:')
print(result[result['alias']=='mlp'].tail(1))

print('neumf metric:')
print(result[result['alias']=='neumf'].tail(1))
