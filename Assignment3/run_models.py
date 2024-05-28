import pandas as pd
import numpy as np
import os
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import csv
# 配置部分
gmf_config = {
    'alias': 'gmf_factor8neg4-implict',
    'num_epoch': 50,
    'batch_size': 1024,
    'optimizer': 'adam',
    'adam_lr': 1e-3,
    'num_users': 6040,
    'num_items': 3706,
    'latent_dim': 8,
    'num_negative': 4,
    'l2_regularization': 0,
    'weight_init_gaussian': True,
    'use_cuda': True,
    'device_id': 0,
    'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
}

mlp_config = {
    'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
    'num_epoch': 50,
    'batch_size': 256,
    'optimizer': 'adam',
    'adam_lr': 1e-3,
    'num_users': 6040,
    'num_items': 3706,
    'latent_dim': 8,
    'num_negative': 4,
    'layers': [16, 64, 32, 16, 8],
    'l2_regularization': 0.0000001,
    'weight_init_gaussian': True,
    'use_cuda': True,
    'device_id': 0,
    'pretrain': False,
    'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
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
                'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.0000001,
                'weight_init_gaussian': True,
                'use_cuda': True,
                'device_id': 0,
                'pretrain': False,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
}


# 加载数据
ml1m_dir = 'data/ml-1m/ratings.dat'
ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')

# 重新索引
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

# 指定模型
# config = gmf_config
# engine = GMFEngine(config)
# config = mlp_config
# engine = MLPEngine(config)
config = neumf_config
engine = NeuMFEngine(config)

# 初始化CSV文件
hr_file = 'hit_ratio.csv'
ndcg_file = 'ndcg.csv'

# 如果文件不存在，则创建文件并写入标题
if not os.path.exists(hr_file):
    with open(hr_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Wall time', 'Step', 'Value'])

if not os.path.exists(ndcg_file):
    with open(ndcg_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Wall time', 'Step', 'Value'])

# 训练和评估
for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    
    # 获取当前时间戳
    wall_time = pd.Timestamp.now().timestamp()
    
    # 将结果追加到CSV文件
    with open(hr_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([wall_time, epoch, hit_ratio])
    
    with open(ndcg_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([wall_time, epoch, ndcg])
