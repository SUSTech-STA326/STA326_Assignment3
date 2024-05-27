import pandas as pd
import numpy as np

def preprocess_data(input_path, output_path):
    # 读取ratings.dat文件
    df = pd.read_csv(input_path, sep='::', header=None, names=['userId', 'itemId', 'rating', 'timestamp'], engine='python')
    df['rating'] = df['rating'].apply(lambda x: 1 if x > 0 else 0)  # 转换为隐式反馈

    # 为每个用户保留最后一次互动作为测试集
    df['rank_latest'] = df.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
    test = df[df['rank_latest'] == 1]
    train = df[df['rank_latest'] != 1]

    # 删除多余的列
    train = train[['userId', 'itemId', 'rating']]
    test = test[['userId', 'itemId', 'rating']]

    # 保存预处理后的数据
    train.to_csv(output_path + 'train.csv', index=False)
    test.to_csv(output_path + 'test.csv', index=False)

if __name__ == "__main__":
    preprocess_data('data/ratings.dat', 'data/')
