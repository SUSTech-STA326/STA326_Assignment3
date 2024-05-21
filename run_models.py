if __name__ == '__main__':
    import sys
    import torch

    print("Python version:")
    print(sys.version)

    print("\nPyTorch version:")
    print(torch.__version__)

    import pandas as pd
    import numpy as np
    from DataBuilder import RatingData, NegativeData
    from Models import GMF, MLP, NeuMF

    # Load the original training data
    df_train = pd.read_csv('./ml-1m/ml-1m.train.rating', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

    # Drop the 'timestamp' column
    df_train.drop(columns='timestamp', inplace=True)

    # Assume total number of items is 3706 (0 to 3705)
    all_items = set(range(3706))

    # Prepare a dictionary to hold negative samples for each user
    negative_samples = {uid: [] for uid in df_train['user_id'].unique()}

    df_train['rating'] = 1

    # Generate negative samples for each user
    for user_id in negative_samples:
        # Get positive samples for this user
        pos_items = set(df_train[df_train['user_id'] == user_id]['item_id'].unique())

        # Calculate the potential negative items
        neg_items = list(all_items - pos_items)

        # Randomly select 8 negative items
        chosen_negatives = np.random.choice(neg_items, 8, replace=False)

        # Append to the negative samples dictionary
        negative_samples[user_id].extend(chosen_negatives)

    # Create a DataFrame for negative samples
    neg_data_list = []
    for user_id, items in negative_samples.items():
        for item in items:
            neg_data_list.append([user_id, item, 0])

    df_negatives = pd.DataFrame(neg_data_list, columns=['user_id', 'item_id', 'rating'])

    # Concatenate the negative samples to the original data
    df_final = pd.concat([df_train, df_negatives], ignore_index=True)
    df_final.to_csv('./ml-1m/ml-1m.train.rating.final', sep='\t', header=False, index=False)

    df_test = pd.read_csv('./ml-1m/ml-1m.test.rating', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df_test.drop(columns='timestamp', inplace=True)
    df_test['rating'] = 1
    df_test.to_csv('./ml-1m/ml-1m.test.rating.final', sep='\t', header=False, index=False)

