import pandas as pd
import numpy as np
from aMLP import MLPEngine
from adata import SampleGenerator
from tqdm import tqdm

mlp_config = {'alias': 'mlp',
              'num_epoch': 10,
              'batch_size': 256, 
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
#               'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
              'layers': [],
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              'model_dir': '/data/lab/HW3/pytorch-version/checkpoint/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

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


layers_=[[16],[16,8],[16,64,8],[16,64,32,8],[16,64,32,16,8]]

for lay in layers_:
    config = mlp_config
    config['layers']=lay
    engine = MLPEngine(config)
    hit_ratio1 =[]
    ndcg1 = []
    for epoch in tqdm(range(config['num_epoch'])):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        hit_ratio1.append(hit_ratio)
        ndcg1.append(ndcg)
        
    # 存结果
    data = {
        'Hit Ratio': hit_ratio1,
        'NDCG': ndcg1
    }
    df = pd.DataFrame(data)
    df.to_csv(f'/data/lab/HW3/pytorch-version/Result/{config["alias"]}_ablation_{(len(config["layers"]))-1}.csv', index=False)