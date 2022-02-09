# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import model
import config
import evaluate_wholeItemsRank
import data_utils
from tqdm import tqdm

from metabalance import MetaBalance


torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--lr",
	type=float,
	default=0.001,
	help="learning rate")
parser.add_argument("--dropout",
	type=float,
	default=0.5,
	help="dropout rate")
parser.add_argument("--batch_size",
	type=int,
	default=256,
	help="batch size for training")
parser.add_argument("--run_name",
	type=str,
	default="log.txt",
	help="name of log of this run")
parser.add_argument("--epochs",
	type=int,
	default=400,
	help="training epoches")
parser.add_argument("--top_k",
	type=int,
	default=20,
	help="compute metrics@top_k")
parser.add_argument("--factor_num",
	type=int,
	default=32,
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers",
	type=int,
	default=3,
	help="number of layers in MLP model")
parser.add_argument("--num_ng",
	type=int,
	default=4,
	help="sample negative items for training")
parser.add_argument("--test_num_ng",
	type=int,
	default=99,
	help="sample part of negative items for testing")
parser.add_argument("--out",
	default=True,
	help="save model or not")
parser.add_argument("--gpu",
	type=str,
	default="0",
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


############################## PREPARE DATASET ##########################
train_data, click_data, fav_data, cart_data, user_num ,item_num, train_mat, click_mat, fav_mat, cart_mat = data_utils.load_all()


# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, click_data, fav_data, cart_data, item_num, train_mat, click_mat, fav_mat, cart_mat, args.num_ng, True)
train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)

########################### CREATE MODEL #################################
if config.model == 'NeuMF-pre':
	assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(config.GMF_model_path)
	MLP_model = torch.load(config.MLP_model_path)
else:
	GMF_model = None
	MLP_model = None

model = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
						args.dropout, config.model, GMF_model, MLP_model)
model.cuda()
loss_function = nn.BCEWithLogitsLoss()


if config.model == 'NeuMF-pre':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
	sharedLayerParameters = list(model.embed_user_GMF.parameters())+list(model.embed_item_GMF.parameters())+list(model.embed_user_MLP.parameters())+list(model.embed_item_MLP.parameters())+list(model.MLP_layers.parameters())

	metabalance = MetaBalance(sharedLayerParameters)
	optimizer_sharedLayer = optim.Adam(sharedLayerParameters, lr=args.lr, weight_decay=0.0000001)

	taskLayerParameters = list(model.predict_layer_buy.parameters())+list(model.predict_layer_click.parameters())+list(model.predict_layer_fav.parameters())+list(model.predict_layer_cart.parameters())
	optimizer_taskLayer = optim.Adam(taskLayerParameters, lr=args.lr, weight_decay=0.0000001)

########################### TRAINING #####################################
f = open(os.path.join(config.log_path, args.run_name), 'w')
count, best_hr = 0, 0

for epoch in range(args.epochs):
	model.train() # Enable dropout (if have).
	start_time = time.time()
	train_loader.dataset.ng_sample()

	#for user, item, label in train_loader:
	print('Training at this epoch start.')
	loss_epoch_buy = 0.0
	loss_epoch_click = 0.0
	loss_epoch_fav = 0.0
	loss_epoch_cart = 0.0
	for step, batch in enumerate(tqdm(train_loader)):
		user = batch[0]
		item = batch[1]
		label = batch[2]
		label_click = batch[3].float().cuda()
		label_fav = batch[4].float().cuda()
		label_cart = batch[5].float().cuda()
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()

		model.zero_grad()

		prediction_buy, prediction_click, prediction_fav, prediction_cart = model(user, item)
		loss_buy = loss_function(prediction_buy, label)
		loss_click = loss_function(prediction_click, label_click)
		loss_fav = loss_function(prediction_fav, label_fav)
		loss_cart = loss_function(prediction_cart, label_cart)

		loss = loss_buy + loss_click + loss_fav + loss_cart
		loss.backward(retain_graph=True)
		optimizer_taskLayer.step()
		model.zero_grad()

		metabalance.step([loss_buy, loss_fav, loss_click, loss_cart])
		optimizer_sharedLayer.step()

		count += 1
		loss_epoch_buy += loss_buy
		loss_epoch_click += loss_click
		loss_epoch_fav += loss_fav
		loss_epoch_cart += loss_cart
	print('buy loss of epoch: '+str(epoch), loss_epoch_buy*1.0/step)
	print('click loss of epoch: '+str(epoch), loss_epoch_click*1.0/step)
	print('fav loss of epoch: '+str(epoch), loss_epoch_fav*1.0/step)
	print('cart loss of epoch: '+str(epoch), loss_epoch_cart*1.0/step)


	model.eval()
	if epoch%10==0 and epoch!=0:
		t_validation = evaluate_wholeItemsRank.evaluateModel(model, user_num, item_num, args.top_k, epoch, 16, config.train_rating, config.validation_rating)
		f.write('Validation in epoch: '+ str(epoch) + ' loss: ' + str(loss_epoch_buy*1.0/step) + '\n' + str(t_validation) + '\n')
		f.flush()

		t_test = evaluate_wholeItemsRank.evaluateModel(model, user_num, item_num, args.top_k, epoch, 16, config.train_rating, config.test_rating)
		f.write('Test in epoch: '+str(epoch)+'\n'+'loss: '+str(loss_epoch_buy*1.0/step)+'\n'+'click loss of epoch: '+str(loss_epoch_click*1.0/step)+'\n'+'fav loss of epoch: '+str(loss_epoch_fav*1.0/step)+'\n'+'cart loss of epoch: '+str(loss_epoch_cart*1.0/step)+'\n'+str(t_test)+'\n')
		f.flush()


		elapsed_time = time.time() - start_time
		print("The time elapse of epoch {:03d}".format(epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

