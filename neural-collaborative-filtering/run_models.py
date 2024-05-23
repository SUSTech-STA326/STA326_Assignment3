import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator

gmf_config = {'alias': 'gmf_factor8neg4-implict',
              'num_epoch': 100,
              'batch_size': 256,#1024
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
              'num_epoch': 100,
              'batch_size': 256,#1024,#256,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16, 64, 32, 16, 8], #[16, 8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'weight_init_gaussian': True,
              'use_cuda': True,
              'device_id': 0,
              'pretrain': False,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'neumf_factor8neg4',
                'num_epoch': 100,
                'batch_size': 256,#1024,#256,
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

mlp_configs = [
    {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001_0 layer', 'num_epoch': 100, 'batch_size': 256, 'optimizer': 'adam', 'adam_lr': 1e-3, 'num_users': 6040, 'num_items': 3706, 'latent_dim': 4, 'num_negative': 4, 'layers': [8], 'l2_regularization': 0.0000001, 'weight_init_gaussian': True, 'use_cuda': True, 'device_id': 0, 'pretrain': False, 'pretrain_mf': 'checkpoints/gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model', 'model_dir': 'checkpoints/mlp_0layer'},
    {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001_1 layer', 'num_epoch': 100, 'batch_size': 256, 'optimizer': 'adam', 'adam_lr': 1e-3, 'num_users': 6040, 'num_items': 3706, 'latent_dim': 8, 'num_negative': 4, 'layers': [16, 8], 'l2_regularization': 0.0000001, 'weight_init_gaussian': True, 'use_cuda': True, 'device_id': 0, 'pretrain': False, 'pretrain_mf': 'checkpoints/gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model', 'model_dir': 'checkpoints/mlp_1layer'},
    {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001_2 layer', 'num_epoch': 100, 'batch_size': 256, 'optimizer': 'adam', 'adam_lr': 1e-3, 'num_users': 6040, 'num_items': 3706, 'latent_dim': 16, 'num_negative': 4, 'layers': [32, 16, 8], 'l2_regularization': 0.0000001, 'weight_init_gaussian': True, 'use_cuda': True, 'device_id': 0, 'pretrain': False, 'pretrain_mf': 'checkpoints/gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model', 'model_dir': 'checkpoints/mlp_2layer'},
    {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001_3 layer', 'num_epoch': 100, 'batch_size': 256, 'optimizer': 'adam', 'adam_lr': 1e-3, 'num_users': 6040, 'num_items': 3706, 'latent_dim': 32, 'num_negative': 4, 'layers': [64, 32, 16, 8], 'l2_regularization': 0.0000001, 'weight_init_gaussian': True, 'use_cuda': True, 'device_id': 0, 'pretrain': False, 'pretrain_mf': 'checkpoints/gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model', 'model_dir': 'checkpoints/mlp_3layer'},
    {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001_4 layer', 'num_epoch': 100, 'batch_size': 256, 'optimizer': 'adam', 'adam_lr': 1e-3, 'num_users': 6040, 'num_items': 3706, 'latent_dim': 64, 'num_negative': 4, 'layers': [128, 64, 32, 16, 8], 'l2_regularization': 0.0000001, 'weight_init_gaussian': True, 'use_cuda': True, 'device_id': 0, 'pretrain': False, 'pretrain_mf': 'checkpoints/gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model', 'model_dir': 'checkpoints/mlp_4layer'}
]

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
GMFengine = GMFEngine(gmf_config)
MLPengine = MLPEngine(mlp_config)
NeuMFengine = NeuMFEngine(neumf_config)

models = [(gmf_config, GMFengine), (mlp_config, MLPengine), (neumf_config, NeuMFengine)]

hr_values_list = []
ndcg_values_list = []

# for config, engine in models:

#     hr_values = []
#     ndcg_values = []
    
#     for epoch in range(config['num_epoch']):
#         print('Training model {} - Epoch {} starts !'.format(config['alias'].split('_')[0], epoch))
#         print('-' * 80)
#         train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
#         engine.train_an_epoch(train_loader, epoch_id=epoch)
#         hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
#         engine.save(config['alias'], epoch, hit_ratio, ndcg)

#         hr_values.append(hit_ratio)
#         ndcg_values.append(ndcg)
    
#     hr_values_list.append(hr_values)
#     ndcg_values_list.append(ndcg_values)

# # 画图
# plt.figure(figsize=(15, 6))

# plt.subplot(1, 2, 1)
# for i, hr_values in enumerate(hr_values_list):
# #     print(range(models[i][0]['num_epoch']))
#     plt.plot(range(models[i][0]['num_epoch']), hr_values, label=models[i][0]['alias'].split('_')[0])
# plt.xlabel('Epoch')
# plt.ylabel('Hit Ratio (HR)')
# plt.title('HR vs. Epoch')
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 2, 2)
# for i, ndcg_values in enumerate(ndcg_values_list):
#     plt.plot(range(models[i][0]['num_epoch']), ndcg_values, label=models[i][0]['alias'].split('_')[0])
# plt.xlabel('Epoch')
# plt.ylabel('NDCG')
# plt.title('NDCG vs. Epoch')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('HR_NDCG_vs_Epoch.png')

# hr_ndcg_data = []

# # 绘制HR和NDCG随着层数变化的图表
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.xlabel('Number of Layers')
# plt.ylabel('Hit Ratio (HR)')
# plt.title('HR vs. Number of Layers')

# plt.subplot(1, 2, 2)
# plt.xlabel('Number of Layers')
# plt.ylabel('NDCG')
# plt.title('NDCG vs. Number of Layers')

# for config in mlp_configs:
#     engine = MLPEngine(config)
   
#     hr_values = []
#     ndcg_values = []
    
#     for epoch in range(config['num_epoch']):
#         print('Training model {} - Epoch {} starts !'.format(config['alias'], epoch))
#         print('-' * 80)
#         train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
#         engine.train_an_epoch(train_loader, epoch_id=epoch)
#         hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
#         engine.save(config['alias'], epoch, hit_ratio, ndcg)

#         hr_values.append(hit_ratio)
#         ndcg_values.append(ndcg)
    
#     hr_ndcg_data.append({'alias': config['alias'], 'HR': hr_values, 'NDCG': ndcg_values})
    
#     # 绘制每个模型的HR和NDCG随着层数变化的图表
#     plt.subplot(1, 2, 1)
#     plt.plot(len(config['layers']), hr_values[-1], 'o', label=f'{config["alias"].split("_")[-1]}')
    
#     plt.subplot(1, 2, 2)
#     plt.plot(len(config['layers']), ndcg_values[-1], 'o', label=f'{config["alias"].split("_")[-1]}')

# # 保存图表
# plt.subplot(1, 2, 1)
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('HR_NDCG_vs_Layers.png')

# # 保存结果到CSV文件
# for data in hr_ndcg_data:
#     df = pd.DataFrame({'HR': data['HR'], 'NDCG': data['NDCG']})
#     df.to_csv(f"{data['alias']}_HR_NDCG.csv", index=False)

# print("HR and NDCG data saved as CSV files.")

file_prefix = 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001_'
file_suffix = ' layer_HR_NDCG.csv'

# 用于存储HR和NDCG数据的列表
hr_data = []
ndcg_data = []
layers = [0, 1, 2, 3, 4]

# 循环读取文件
for layer in layers:
    file_name = f'{file_prefix}{layer}{file_suffix}'
    data = pd.read_csv(file_name)
#     print(data)
    hr_data.append((layer, data['HR']))
    ndcg_data.append((layer, data['NDCG']))

plt.figure(figsize=(15, 6))

# HR
plt.subplot(1, 2, 1)
for layer, hr_values in hr_data:
    plt.plot(hr_values.index, hr_values, label=f'{layer} layer')
plt.xlabel('Epoch')
plt.ylabel('Hit Ratio (HR)')
plt.title('HR vs. Epoch')
plt.legend()
plt.grid(True)

# NDCG
plt.subplot(1, 2, 2)
for layer, ndcg_values in ndcg_data:
    plt.plot(ndcg_values.index, ndcg_values, label=f'{layer} layer')
plt.xlabel('Epoch')
plt.ylabel('NDCG')
plt.title('NDCG vs. Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('HR_NDCG_vs_Epoch_layers.png')
#plt.show()
