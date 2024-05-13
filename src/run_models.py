import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import os
import matplotlib.pyplot as plt

# 创建 checkpoints 目录如果它不存在
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
    
gmf_config = {'alias': 'gmf_factor8neg4-implict',
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
              'weight_init_gaussian': True,
              'use_cuda': True,
              'device_id': 0,
              'pretrain': False,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

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
# config = gmf_config
# engine = GMFEngine(config)
# config = mlp_config
# engine = MLPEngine(config)
# config = neumf_config
# engine = NeuMFEngine(config)
# for epoch in range(config['num_epoch']):
#     print('Epoch {} starts !'.format(epoch))
#     print('-' * 80)
#     train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
#     engine.train_an_epoch(train_loader, epoch_id=epoch)
#     hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
#     engine.save(config['alias'], epoch, hit_ratio, ndcg)



engine_gmf = GMFEngine(gmf_config)
engine_mlp = MLPEngine(mlp_config)
engine_neumf = NeuMFEngine(neumf_config)

# 训练和评估三个模型
for epoch in range(gmf_config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)

    # 训练和评估GMF模型
    train_loader = sample_generator.instance_a_train_loader(gmf_config['num_negative'], gmf_config['batch_size'])
    engine_gmf.train_an_epoch(train_loader, epoch_id=epoch)
    engine_gmf.evaluate(evaluate_data, epoch_id=epoch)

for epoch in range(mlp_config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)

    # 训练和评估MLP模型
    train_loader = sample_generator.instance_a_train_loader(mlp_config['num_negative'], mlp_config['batch_size'])
    engine_mlp.train_an_epoch(train_loader, epoch_id=epoch)
    engine_mlp.evaluate(evaluate_data, epoch_id=epoch)

for epoch in range(neumf_config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)

    # 训练和评估NeuMF模型
    train_loader = sample_generator.instance_a_train_loader(neumf_config['num_negative'], neumf_config['batch_size'])
    engine_neumf.train_an_epoch(train_loader, epoch_id=epoch)
    engine_neumf.evaluate(evaluate_data, epoch_id=epoch)

# 创建一个文件夹来存储图片
if not os.path.exists('images'):
    os.makedirs('images')

# 绘制HR@10图
plt.figure(figsize=(10, 8))
plt.plot(range(gmf_config['num_epoch']), engine_gmf.hr_list, color='darkblue', linestyle='--', label='GMF')
plt.plot(range(mlp_config['num_epoch']), engine_mlp.hr_list, color='darkmagenta', linestyle='-', label='MLP')
plt.plot(range(neumf_config['num_epoch']), engine_neumf.hr_list, color='red', linestyle='-', label='NeuMF')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('HR@10', fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.savefig('images/HR@10_epoch.png', dpi=300)

# 绘制NDCG@10图
plt.figure(figsize=(10, 8))
plt.plot(range(gmf_config['num_epoch']), engine_gmf.ndcg_list, color='darkblue', linestyle='--', label='GMF')
plt.plot(range(mlp_config['num_epoch']), engine_mlp.ndcg_list, color='darkmagenta', linestyle='-', label='MLP')
plt.plot(range(neumf_config['num_epoch']), engine_neumf.ndcg_list, color='red', linestyle='-', label='NeuMF')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('NDCG@10', fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.savefig('images/NDCG@10_epoch.png', dpi=300)

# 绘制训练损失图
plt.figure(figsize=(10, 8))
plt.plot(range(gmf_config['num_epoch']), engine_gmf.train_loss, color='darkblue', linestyle='--', label='GMF')
plt.plot(range(mlp_config['num_epoch']), engine_mlp.train_loss, color='darkmagenta', linestyle='-', label='MLP')
plt.plot(range(neumf_config['num_epoch']), engine_neumf.train_loss, color='red', linestyle='-', label='NeuMF')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Training Loss', fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.savefig('images/trainingloss_epoch.png', dpi=300)


print(f"After {gmf_config['num_epoch']} epoch:")
print("hr_list_gmf:", engine_gmf.hr_list[-1])
print("hr_list_mlp:", engine_mlp.hr_list[-1])
print("hr_list_neumf:", engine_neumf.hr_list[-1])
print("ndcg_list_gmf:", engine_gmf.ndcg_list[-1])
print("ndcg_list_mlp:", engine_mlp.ndcg_list[-1])
print("ndcg_list_neumf:", engine_neumf.ndcg_list[-1])

