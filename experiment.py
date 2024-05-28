import os
import time
import argparse
import pandas as pd
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


import model as Model
import config 
import util
import data_utils
import evaluate



def experiment(method='NeuMF', 
               seed=42,
               lr=0.001,
               dropout=0.2,
               batch_size=256,
               epochs=30,
               top_k=10,
               embed_size=32,
               layers=[64,32,16,8],
               num_ng=4,
               num_ng_test=100,
               out=True):
    
	# set device
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	util.seed_everything(seed)






	# load data
	class RequiredArgs:
		def __init__(self, method, seed, lr, dropout, batch_size, epochs, top_k, embed_size, layers, num_ng, num_ng_test, out):
			self.method = method
			self.seed = seed
			self.lr = lr
			self.dropout = dropout
			self.batch_size = batch_size
			self.epochs = epochs
			self.top_k = top_k
			self.embed_size = embed_size
			self.layers = layers
			self.num_ng = num_ng
			self.num_ng_test = num_ng_test
			self.out = out

	args = RequiredArgs(method, seed, lr, dropout, batch_size, epochs, top_k, embed_size, layers, num_ng, num_ng_test, out)
	ml_1m = pd.read_csv(
		config.DATA_PATH, 
		sep="::", 
		names = ['user_id', 'item_id', 'rating', 'timestamp'], 
		engine='python')

	num_users = ml_1m['user_id'].nunique()+1
	num_items = ml_1m['item_id'].nunique()+1

	data = data_utils.NCF_Data(args, ml_1m)
	train_loader =data.get_train_instance()
	test_loader =data.get_test_instance()


 
 
 
 	# model, loss function, optimizer
	if method == "GMF": 
		model = Model.Generalized_Matrix_Factorization(args, num_users, num_items)
	if method == "MLP":
		model = Model.Multi_Layer_Perceptron(args, num_users, num_items)
	if method == "NeuMF": 
		model = Model.NeuMF(args, num_users, num_items)
	model = model.to(device)
	loss_function = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)






	# train, evaluation
	best_hr = 0
	best_ndcg = 0
	for epoch in range(1, args.epochs+1):
		model.train()
		start_time = time.time()

		for user, item, label in train_loader:
			user = user.to(device)
			item = item.to(device)
			label = label.to(device)

			optimizer.zero_grad()
			prediction = model(user, item)

			if args.method == 'GMF' or args.method == 'MLP':
				prediction = prediction.squeeze()

			loss = loss_function(prediction, label)
			loss.backward()
			optimizer.step()

		model.eval()
		HR, NDCG = evaluate.metrics(model, test_loader, args.top_k, device)
		print("HR = {:.3f}, NDCG = {:.3f}".format(HR, NDCG))
		if HR > best_hr: best_hr = HR
		if NDCG > best_ndcg: best_ndcg = NDCG
	
 

	print("Best HR = {:.3f}, NDCG = {:.3f}".format(best_hr, best_ndcg))
	return best_hr, best_ndcg
