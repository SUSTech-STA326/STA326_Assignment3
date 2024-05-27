import pandas as pd
import numpy as np
from models.gmf import GMFEngine
from models.mlp import MLPEngine
from models.neumf import NeuMFEngine
from tools.data_loader import SampleGenerator
import os
import matplotlib.pyplot as plt

gmf_config = {'alias': 'gmf_factor8neg4-implict',
                'num_epoch': 50,
                'batch_size': 2048,
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
              'batch_size': 2048,
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
                'batch_size': 2048,
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
ml1m_dir = '../data/ml-1m/ratings.dat'
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




# Define different MLP layers
mlp_layers = [[16], [16, 8], [16, 32, 8], [16, 32, 16, 8], [16, 64, 32, 16, 8]]

results = {
    'Model': [],
    'Epoch': [],
    'HR@10': [],
    'NDCG@10': []
}

for idx, mlp_layer in enumerate(mlp_layers):
    mlp_config['layers'] = mlp_layer
    mlp_config['alias'] = 'mlp_factor8neg4_bz256_{}_pretrain_reg_0.0000001'.format('-'.join(str(e) for e in mlp_layer))
    mlp_config['num_epoch'] = 30
    engine = MLPEngine(mlp_config)
    for epoch in range(mlp_config['num_epoch']):
        print(f'MLP{mlp_layer} Epoch {epoch} starts!')
        print('-' * 80)
        # 训练模型
        train_loader = sample_generator.instance_a_train_loader(mlp_config['num_negative'], mlp_config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        engine.evaluate(evaluate_data, epoch_id=epoch)
        
        results['Model'].append(f'MLP-{mlp_layer}')
        results['Epoch'].append(epoch + 1)  
        results['HR@10'].append(engine.hr_list[-1]) 
        results['NDCG@10'].append(engine.ndcg_list[-1])  

    # 将每个配置的HR和NDCG保存为单独的CSV文件
    data = {
        'Hit Ratio': engine.hr_list,
        'NDCG': engine.ndcg_list
    }
    df = pd.DataFrame(data)
    df.to_csv('images/mlp_metrics_{}.csv'.format(idx), index=False)

# 保存所有结果到一个总的CSV文件
df_results = pd.DataFrame(results)
df_results.to_csv('images/mlp_metrics_comparison.csv', index=False)

# 绘制不同MLP层配置的HR@10和NDCG@10曲线
plt.figure(figsize=(10, 8))
for key in set(results['Model']):
    model_results = df_results[df_results['Model'] == key]
    plt.plot(model_results['Epoch'], model_results['HR@10'], label=f'{key} HR@10')
plt.title('HR@10 for Different MLP Layers')
plt.xlabel('Epoch')
plt.ylabel('HR@10')
plt.legend()
plt.grid(True)
plt.savefig('images/mlp_hr_comparison.png')
plt.show()


# Comparison Experiment
engine_gmf = GMFEngine(gmf_config)
engine_mlp = MLPEngine(mlp_config)
engine_neumf = NeuMFEngine(neumf_config)
# 定义模型引擎列表和配置字典列表
engines = [engine_gmf, engine_mlp, engine_neumf]
configs = [gmf_config, mlp_config, neumf_config]

# 循环训练和评估每个模型
for engine, config in zip(engines, configs):
    for epoch in range(config['num_epoch']):
        print(f'Epoch {epoch} starts !')
        print('-' * 80)

        # 训练模型
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        engine.evaluate(evaluate_data, epoch_id=epoch)

if not os.path.exists('./images'):
    os.makedirs('./images')

    # 训练模型
    train_loader = sample_generator.instance_a_train_loader(mlp_config['num_negative'], mlp_config['batch_size'])
    engine_mlp.train_an_epoch(train_loader, epoch_id=epoch)
    engine_mlp.evaluate(evaluate_data, epoch_id=epoch)
    
data_gmf = {
    'Hit Ratio': engine_gmf.hr_list,
    'NDCG': engine_gmf.ndcg_list
}
df_gmf = pd.DataFrame(data_gmf)
df_gmf.to_csv('images/gmf_metrics.csv', index=False)

data_mlp = {
    'Hit Ratio': engine_mlp.hr_list,
    'NDCG': engine_mlp.ndcg_list
}
df_mlp = pd.DataFrame(data_mlp)
df_mlp.to_csv('images/mlp_metrics.csv', index=False)

data_neumf = {
    'Hit Ratio': engine_neumf.hr_list,
    'NDCG': engine_neumf.ndcg_list
}
df_neumf = pd.DataFrame(data_neumf)
df_neumf.to_csv('images/neumf_metrics.csv', index=False)

data_loss = {
    'GMF': engine_gmf.train_loss,
    'MLP': engine_mlp.train_loss,
    'NeuMF': engine_neumf.train_loss
}
df_loss = pd.DataFrame(data_loss)
df_loss.to_csv('images/loss.csv', index=False)

print(f"After {gmf_config['num_epoch']} epoch:")
print("gmf-HR:", engine_gmf.hr_list[-1])
print("mlp-HR:", engine_mlp.hr_list[-1])
print("neumf-HR:", engine_neumf.hr_list[-1])
print("gmf-NDCG:", engine_gmf.ndcg_list[-1])
print("mlp-NDCG:", engine_mlp.ndcg_list[-1])
print("neumf-NDCG:", engine_neumf.ndcg_list[-1])


# # 画图
# plt.figure(figsize=(10, 8))
# plt.plot(range(gmf_config['num_epoch']), engine_gmf.hr_list, color='#ADD8E6', linestyle='--', label='GMF-HR')
# plt.plot(range(mlp_config['num_epoch']), engine_mlp.hr_list, color='#845EC2', linestyle='-', label='MLP-HR')
# plt.plot(range(neumf_config['num_epoch']), engine_neumf.hr_list, color='#FF9999', linestyle='-', label='NeuMF-HR')
# plt.xlabel('Epoch', fontsize=14)
# plt.ylabel('HR@10', fontsize=14)
# plt.legend(fontsize=14)
# plt.grid(True)
# plt.savefig('images/HR@10_epoch.png', dpi=300)

# plt.figure(figsize=(10, 8))
# plt.plot(range(gmf_config['num_epoch']), engine_gmf.ndcg_list, color='#ADD8E6', linestyle='--', label='GMF-NDCG')
# plt.plot(range(mlp_config['num_epoch']), engine_mlp.ndcg_list, color='#845EC2', linestyle='-', label='MLP-NDCG')
# plt.plot(range(neumf_config['num_epoch']), engine_neumf.ndcg_list, color='#FF9999', linestyle='-', label='NeuMF-NDCG')
# plt.xlabel('Epoch', fontsize=14)
# plt.ylabel('NDCG@10', fontsize=14)
# plt.legend(fontsize=14)
# plt.grid(True)
# plt.savefig('images/NDCG@10_epoch.png', dpi=300)

# plt.figure(figsize=(10, 8))
# plt.plot(range(gmf_config['num_epoch']), engine_gmf.train_loss, color='#ADD8E6', linestyle='--', label='GMF-loss')
# plt.plot(range(mlp_config['num_epoch']), engine_mlp.train_loss, color='#845EC2', linestyle='-', label='MLP-loss')
# plt.plot(range(neumf_config['num_epoch']), engine_neumf.train_loss, color='#FF9999', linestyle='-', label='NeuMF-loss')
# plt.xlabel('Epoch', fontsize=14)
# plt.ylabel('Training Loss', fontsize=14)
# plt.legend(fontsize=14)
# plt.grid(True)
# plt.savefig('images/trainingloss_epoch.png', dpi=300)



