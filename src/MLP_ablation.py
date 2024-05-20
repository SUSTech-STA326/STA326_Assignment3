import pandas as pd
import numpy as np
from mlp import MLPEngine
from data import SampleGenerator
import os
import matplotlib.pyplot as plt

# 创建必要的目录
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('mlp_ablation', exist_ok=True)
os.makedirs('images', exist_ok=True)

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

# DataLoader for training
sample_generator = SampleGenerator(ratings=ml1m_rating)
evaluate_data = sample_generator.evaluate_data

# 定义不同的MLP配置
mlp_configs = [
    {'alias': 'mlp_0', 'layers': [16]},                       
    {'alias': 'mlp_1', 'layers': [16, 8]},                    
    {'alias': 'mlp_2', 'layers': [16, 32, 8]},              
    {'alias': 'mlp_3', 'layers': [16, 32, 16, 8]},        
    {'alias': 'mlp_4', 'layers': [16, 64, 32, 16, 8]},  
]
# 存储每个MLP配置的引擎
mlp_engines = []

# 训练和评估不同的MLP配置
for mlp_config in mlp_configs:
    config = {
        'alias': mlp_config['alias'],
        'num_epoch': 50,
        'batch_size': 256,
        'optimizer': 'adam',
        'adam_lr': 1e-3,
        'num_users': 6040,
        'num_items': 3706,
        'latent_dim': 8,
        'num_negative': 4,
        'layers': mlp_config['layers'],
        'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
        'weight_init_gaussian': True,
        'use_cuda': True,
        'device_id': 0,
        'pretrain': False,
        'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
        'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
    }
    
    print(f"Training MLP with config: {config['alias']}")
    engine = MLPEngine(config)
    mlp_engines.append(engine)
    
    for epoch in range(config['num_epoch']):
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        engine.evaluate(evaluate_data, epoch_id=epoch)
        

# 绘制图像
for metric, key in [('HR@10', 'hr_list'), ('NDCG@10', 'ndcg_list'), ('Training Loss', 'train_loss')]:
    plt.figure(figsize=(10, 8))
    for engine in mlp_engines:
        plt.plot(range(engine.config['num_epoch']), getattr(engine, key), label=engine.config['alias'])
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(f"images/mlp_{metric.lower().replace(' ', '_')}_epoch.png", dpi=300)

# 保存评估结果到文件
with open('mlp_ablation/mlp_data.txt', 'w') as f:
    for engine in mlp_engines:
        f.write(f"Config: {engine.config['alias']}\n")
        f.write(f"HR@10: {engine.hr_list[-1]}\n")
        f.write(f"NDCG@10: {engine.ndcg_list[-1]}\n")
        f.write(f"Training Loss: {engine.train_loss[-1]}\n\n")