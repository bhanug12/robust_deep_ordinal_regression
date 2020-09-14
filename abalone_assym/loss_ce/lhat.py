#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch 
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import time
import argparse 
import sys
from sklearn.model_selection import  StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# code to run multiple times, and code to generate noise 

parser = argparse.ArgumentParser()
parser.add_argument("-h_l",dest='hidden_layer' ,help="hidden layer size", type=int, nargs='+')
parser.add_argument("-lr",dest='l_rate' ,help="learning rate", type=float, nargs='+')
parser.add_argument("-n", dest="noise_model", help="noise rate", type=int)
args = parser.parse_args()


lrate = args.l_rate[0]
hd_sz = args.hidden_layer[0]
ind = args.noise_model


#pre processing
df = pd.read_csv('../abalone.data', header=None)

def gender(a):
    if a=='M':
        return 0
    elif a=='F':
        return 1
    else :
        return 2
label = 8
def buck(a): # 1-7, 8-9, 10-12, 13-29
    if a<=7:
        return 1
    elif a<=9:
        return 2
    elif a<=12:
        return 3
    else :
        return 4

df[0] = df[0].map(gender)
df[8] = df[8].map(buck)
#df.rename(columns={8:'y'}, inplace=True)

for i in range(0,8):
    m = df[i].mean()
    sd = df[i].std()
    df[i] = (df[i] - m)/sd



# Hyper-parameters 
input_size = 8
#hidden_size = 8
label = 8
num_classes = 4
num_epochs = 200
batch_size = 16
#learning_rate = 0.005
validation_split = .2
shuffle_dataset = True
random_seed = 1
LOGFILE = str(ind)+'/log'


class CustomDataset(Dataset):
    def __init__(self, df, label, num_classes):
        self.df = df.copy()
        self.labels = self.df[label]
        self.df.drop(columns=[label], inplace=True)
        self.data_len = len(self.df.index)
        self.nc = num_classes

    def __getitem__(self, index):
        l = pd.DataFrame(self.labels).values[index]
        d = torch.tensor(self.df.iloc[index, :].values)
        return (d,torch.tensor(l,dtype=torch.long))

    def __len__(self):
        return self.data_len

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1 ,num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        #self.fcn = nn.Linear(hidden_size1, hidden_size2)
        #self.relu1 = nn.ReLU()
        self.fcn = nn.Linear(hidden_size1,1,bias=False)
        self.linear_l = nn.Linear(1,num_classes-1)
        self.linear_l.weight = nn.Parameter(torch.ones(num_classes-1,1), requires_grad=False)
        self.linear_l.bias = nn.Parameter(torch.zeros(num_classes-1), requires_grad=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fcn(out)
        #out = self.relu1(out)
        #out = self.fc2(out)
        logits = self.linear_l(out)
        probs = self.sig(logits)
        return logits, probs

def l_hat(logits, l, eta_inv, num_classes):
    los = torch.zeros_like(logits).cuda()
    for i in range(num_classes):
        t = torch.tensor([1]*(i)+[0]*(num_classes-i-1), dtype=torch.double, device='cuda')
        los += -(eta_inv[l-1][:,:,i] * (t*F.logsigmoid(logits+1e-10) + (1.0-t)*torch.log(1.0+1e-10-torch.sigmoid(logits))))
    return torch.mean(torch.sum(los, dim=1))


def compute_mae_and_mse(model, data_loader, device):
    mae, mse, mzo, num_examples = 0, 0, 0, 0
    for i, (features,  levels) in enumerate(data_loader):
        features = features.to(device)
        levels = levels.to(device)
        logits, probas = model(features)
        probas = probas.to(torch.float64)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1) + 1
        num_examples += levels.size(0)
        mae += torch.sum(torch.abs(predicted_labels - levels.view(-1)))
        mse += torch.sum((predicted_labels - levels.view(-1))**2)
        mzo += torch.sum(predicted_labels!=levels.view(-1))
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    mzo = mzo.float() / num_examples
    return mae, mse, mzo, num_examples





def reversals(a):
	for i in range(a.size()[0] - 1):
		if a[i] < a[i+1]:
			return 1 
	return 0 




####eta and eta inverse
eta = np.zeros((num_classes,num_classes), dtype=float)
for i in range(num_classes):
    for j in range(num_classes):
        if i!=j:
            eta[i][j] = 0

for i in range(num_classes):
    eta[i][i] = 1 
    
eta_inv = np.linalg.inv(eta)
eta_inv = torch.from_numpy(eta_inv)
eta_inv = eta_inv.to(device)



#eta for noise
#eta_n = np.zeros((num_classes,num_classes), dtype=float)
#for i in range(num_classes):
#	for j in range(num_classes):
#		if i!=j:
#			if ind==0:
#				eta_n[i][j] = 0
#			elif ind==1:
#				eta_n[i][j] = 0.05/abs(i-j)
#			elif ind==2:
#				eta_n[i][j] = 0.1/abs(i-j)
#			elif ind==3:
#				eta_n[i][j] = 0.15/abs(i-j)
#			elif ind==4:
#				eta_n[i][j] = np.exp(-3*abs(i-j))
#			elif ind==5:
#				eta_n[i][j] = np.exp(-2.2*abs(i-j))
#			elif ind==6:
#				eta_n[i][j] = np.exp(-1.8*abs(i-j))
#			
#
#for i in range(num_classes):
#    eta_n[i][i] = 1 - sum(eta_n)[i]
if ind==0:
        eta_n = eta
elif ind==3:
        eta_n = np.array([[.7, .15,.15,0], [.0,.8,.15,.05], [0.02,.2,.78,.0],[.05,0.1,.15,.7]])
    
eta_inv_n = np.linalg.inv(eta_n)
eta_inv_n = torch.from_numpy(eta_inv_n)
eta_inv_n = eta_inv_n.to(device)

eta_n_s = eta_n.cumsum(axis=1)

def noii(true_y):
    a = np.random.uniform()
    for i in range(num_classes):
        if a <= eta_n_s[true_y-1][i]:
            break
    return i+1






#dataset_size = len(dat)
#indices = list(range(dataset_size))
#split = int(np.floor(validation_split * dataset_size))
#if shuffle_dataset :
#    np.random.seed(random_seed)
#    np.random.shuffle(indices)
#train_indices, val_indices = indices[split:], indices[:split]
#from torch.utils.data.sampler import SubsetRandomSampler
#train_sampler = SubsetRandomSampler(train_indices)
#test_sampler = SubsetRandomSampler(val_indices)
#dat_train = torch.utils.data.DataLoader(dataset=dat,batch_size=batch_size,sampler=train_sampler)
#dat_test = torch.utils.data.DataLoader(dataset=dat,batch_size=batch_size,sampler=test_sampler)

mse_tr_f = []
mae_tr_f = []
mzo_tr_f = []
mse_te_f = []
mae_te_f = []
mzo_te_f = []

mse_tr_b_f = []
mae_tr_b_f = []
mzo_tr_b_f = []
mse_te_b_f = []
mae_te_b_f = []
mzo_te_b_f = []


mse_tr_hat_f = []
mae_tr_hat_f = []
mzo_tr_hat_f = []
mse_te_hat_f = []
mae_te_hat_f = []
mzo_te_hat_f = []



dat = CustomDataset(df, label, num_classes) 
fold = 0
seed_range = [i for i in range(10)] 

for rseed in seed_range:
	mse_tr = []
	mae_tr = []
	mzo_tr = []
	mse_te = []
	mae_te = []
	mzo_te = []
	
	mse_tr_b = []
	mae_tr_b = []
	mzo_tr_b = []
	mse_te_b = []
	mae_te_b = []
	mzo_te_b = []

	mse_tr_hat = []
	mae_tr_hat = []
	mzo_tr_hat = []
	mse_te_hat = []
	mae_te_hat = []
	mzo_te_hat = []


	np.random.seed(rseed)
	dataset_size = df.shape[0]
	indices = list(range(dataset_size))
	np.random.shuffle(indices)
	split = int(np.floor(validation_split * dataset_size))
	train_indices, val_indices = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_indices)
	test_sampler = SubsetRandomSampler(val_indices)
	tr_df = df.iloc[train_indices]
	te_df = df.iloc[val_indices]
	np.random.seed(rseed)	
	tr_df[label] = tr_df[label].map(noii)
	#w_noise_df = df.drop(columns=[label+'_n'])
	#noise_df = df.drop(columns=[label]).rename(columns={label+'_n' : label})
	
	dat_wn = CustomDataset(te_df, label, num_classes) 
	dat_n = CustomDataset(tr_df, label, num_classes) 

	dat_train = torch.utils.data.DataLoader(dataset=dat_n, batch_size=batch_size, shuffle=True)
	dat_test = torch.utils.data.DataLoader(dataset=dat_wn, batch_size=batch_size, shuffle=True)
	#dat_train_wn = torch.utils.data.DataLoader(dataset=dat_wn, batch_size=batch_size, shuffle=True)
	
	torch.manual_seed(rseed)
	model = NeuralNet(input_size=input_size, hidden_size1=hd_sz, num_classes=num_classes)


#	torch.manual_seed(rseed)
#	model_b = NeuralNet(input_size=input_size, hidden_size1=hd_sz, num_classes=num_classes)
#	
#	torch.manual_seed(rseed)
#	model_hat = NeuralNet(input_size=input_size, hidden_size1=hd_sz, num_classes=num_classes)
#

	model.to(device, dtype=torch.float64)
#	model_b.to(device, dtype=torch.float64)
#	model_hat.to(device, dtype=torch.float64)
	fold += 1
	start_time = time.time()
	final_thresh = 0 

#####################for l - hat 
	count_iter = 0
	irreg = 0
	for epoch in range(int(num_epochs)):
		model.train()
		running_loss = 0.0
		total_rows = 0
		lrate = args.l_rate[0]
		if epoch%30==0 & epoch > 0:
		    lrate/=2
	  
		optimizer = torch.optim.AdamW(model.parameters(), lr=lrate)
		for batch_idx, (features, l) in enumerate(dat_train):
			features = features.to(device)
			l = l.to(device)
			logits, probas = model(features)
			cost = l_hat(logits, l, eta_inv_n, num_classes)
			optimizer.zero_grad()
			cost.backward()
			optimizer.step()
			count_iter += 1
		#	params = model.state_dict()
			irreg += reversals(model.state_dict()['linear_l.bias'])
		#	a = params['linear_l.bias']
		#	for i in range(a.size()[0] - 1):
		#		if a[i] < a[i+1]:
		#			irreg += 1
		#			break
			total_rows += l.size(0)
			running_loss += cost.item()*l.size(0)
		s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
		ls = 'Epochs=%f, Total Loss=%.3f, Loss(per_example)=%.3f, Examples=%f' % (epoch, running_loss,running_loss/total_rows,total_rows)
		print('%s \n%s\n' % (ls,s))
		with open(LOGFILE+'_N_'+str(rseed)+'_', 'a') as f:
			f.write('%s \n%s\n' % (ls,s))
	
		model = model.to(torch.float64)
		model.eval()
		with torch.set_grad_enabled(False):  # save memory during inference
		#	train_mae, train_mse, train_mzo, ex1 = compute_mae_and_mse(model, dat_train,device=device)
			test_mae, test_mse, test_mzo, ex2 = compute_mae_and_mse(model, dat_test,device=device)
		#	s = 'Train - MAE/RMSE/MZO/EX: %.2f/%.2f/%.2f/%.2f \n ' % (train_mae, torch.sqrt(train_mse), train_mzo, ex1)
		#	print(s)
			with open(LOGFILE+'_N_'+str(rseed)+'_', 'a') as f:
				f.write('%f' % epoch)
		#		f.write('\n%s\n' % s)
			
			s = 'Test - MAE/RMSE/MZO/EX:  %.2f/%.2f/%.2f/%.2f \n' % (test_mae, torch.sqrt(test_mse), test_mzo, ex2)
			print(s)
			with open(LOGFILE+'_N_'+str(rseed)+'_', 'a') as f:
				f.write('%s\n' % s)
						    		    
#########out final results of model	
	


	final_thresh = reversals(model.state_dict()['linear_l.bias'])
	model = model.to(torch.float64)
	model.eval()
	with torch.set_grad_enabled(False):  # save memory during inference
		train_mae, train_mse, train_mzo, ex1 = compute_mae_and_mse(model, dat_train,device=device)
		test_mae, test_mse, test_mzo, ex2 = compute_mae_and_mse(model, dat_test,device=device)
		s = 'Train - MAE/RMSE/MZO/EX: %.2f/%.2f/%.2f/%.2f \n ' % (train_mae, torch.sqrt(train_mse), train_mzo, ex1)
		print(s)
		with open(LOGFILE+'_N_'+str(rseed)+'_', 'a') as f:
			f.write('%f\n%f\n%s\n' %(rseed,fold ,s))
		with open(str(ind)+'/all_resultslhkr', 'a') as f:
			f.write(LOGFILE+'_N_'+str(rseed)+'_'+str(fold)+'\n%f   %f\n%s\n' % (rseed, fold, s))
		
		s = 'Test - MAE/RMSE/MZO/EX:  %.2f/%.2f/%.2f/%.2f \n' % (test_mae, torch.sqrt(test_mse), test_mzo, ex2)
		print(s)
		with open(LOGFILE+'_N_'+str(rseed)+'_', 'a') as f:
		    f.write('%s\n' % s)
		with open(str(ind)+'/all_resultslhkr', 'a') as f:
		    f.write('%s\n' % s)


		mse_tr_hat.append(np.array(torch.sqrt(train_mse).cpu()))
		mae_tr_hat.append(np.array(train_mae.cpu()))
		mzo_tr_hat.append(np.array(train_mzo.cpu()))
		mse_te_hat.append(np.array(torch.sqrt(test_mse).cpu()))
		mae_te_hat.append(np.array(test_mae.cpu()))
		mzo_te_hat.append(np.array(test_mzo.cpu()))
				


		mse_tr_hat_f.append(np.array(torch.sqrt(train_mse).cpu()))
		mae_tr_hat_f.append(np.array(train_mae.cpu()))
		mzo_tr_hat_f.append(np.array(train_mzo.cpu()))
		mse_te_hat_f.append(np.array(torch.sqrt(test_mse).cpu()))
		mae_te_hat_f.append(np.array(test_mae.cpu()))
		mzo_te_hat_f.append(np.array(test_mzo.cpu()))


		
	

#	with open('model_b'+'_'+str(rseed)+'_'+str(fold)+'.pt', 'wb') as f:
#				torch.save(model_b, f)
#	
	with open(str(ind)+'/model'+'_'+str(rseed)+'_'+str(fold)+'.pt', 'wb') as f:
		torch.save(model, f)
#			
#	with open('model'+'_'+str(rseed)+'_'+str(fold)+'.pt', 'wb') as f:
#				torch.save(model_hat, f)


#with open('all_results_f', 'a') as f:
#	f.write('\n'+str(rseed)+'_'+str(fold)+'_model\n')
#	f.write('Train -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_tr).mean(),np.array(mae_tr).mean(), np.array(mzo_tr).mean() ) )
#	f.write('Test -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_te).mean(),np.array(mae_te).mean(), np.array(mzo_te).mean() ) )
#	f.write('Train _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_tr).std(),np.array(mae_tr).std(), np.array(mzo_tr).std() ) )
#	f.write('Test _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_te).std(),np.array(mae_te).std(), np.array(mzo_te).std() ) )
#
#	f.write('\nmodel_hat\n')
#	f.write('Train -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_tr_hat).mean(),np.array(mae_tr_hat).mean(), np.array(mzo_tr_hat).mean() ) )
#	f.write('Test -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_te_hat).mean(),np.array(mae_te_hat).mean(), np.array(mzo_te_hat).mean() ) )
#	f.write('Train _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_tr_hat).std(),np.array(mae_tr_hat).std(), np.array(mzo_tr_hat).std() ) )
#	f.write('Test _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_te_hat).std(),np.array(mae_te_hat).std(), np.array(mzo_te_hat).std() ) )
#
#
#	f.write('\nmodel_b\n')
#	f.write('Train -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_tr_b).mean(),np.array(mae_tr_b).mean(), np.array(mzo_tr_b).mean() ) )
#	f.write('Test -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_te_b).mean(),np.array(mae_te_b).mean(), np.array(mzo_te_b).mean() ) )
#	f.write('Train _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_tr_b).std(),np.array(mae_tr_b).std(), np.array(mzo_tr_b).std() ) )
#	f.write('Test _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_te_b).std(),np.array(mae_te_b).std(), np.array(mzo_te_b).std() ) )
#
	with open(str(ind)+'/all_results_of_lhkr', 'a') as f:
		f.write('_model\n')
#		f.write('Train -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_tr_f).mean(),np.array(mae_tr_f).mean(), np.array(mzo_tr_f).mean() ) )
#		f.write('Test -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_te_f).mean(),np.array(mae_te_f).mean(), np.array(mzo_te_f).mean() ) )
#		f.write('Train _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_tr_f).std(),np.array(mae_tr_f).std(), np.array(mzo_tr_f).std() ) )
#		f.write('Test _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_te_f).std(),np.array(mae_te_f).std(), np.array(mzo_te_f).std() ) )
#	
#		f.write('model_hat\n')
		f.write('Train -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_tr_hat_f).mean(),np.array(mae_tr_hat_f).mean(), np.array(mzo_tr_hat_f).mean() ) )
		f.write('Test -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_te_hat_f).mean(),np.array(mae_te_hat_f).mean(), np.array(mzo_te_hat_f).mean() ) )
		f.write('Train _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_tr_hat_f).std(),np.array(mae_tr_hat_f).std(), np.array(mzo_tr_hat_f).std() ) )
		f.write('Test _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_te_hat_f).std(),np.array(mae_te_hat_f).std(), np.array(mzo_te_hat_f).std() ) )
	
#		f.write('model_b\n')
#		f.write('Train -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_tr_b_f).mean(),np.array(mae_tr_b_f).mean(), np.array(mzo_tr_b_f).mean() ) )
#		f.write('Test -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_te_b_f).mean(),np.array(mae_te_b_f).mean(), np.array(mzo_te_b_f).mean() ) )
#		f.write('Train _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_tr_b_f).std(),np.array(mae_tr_b_f).std(), np.array(mzo_tr_b_f).std() ) )
#		f.write('Test _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_te_b_f).std(),np.array(mae_te_b_f).std(), np.array(mzo_te_b_f).std() ) )
#	
	
	with open(str(ind)+'/irrev'+'lhkr', 'a') as f:
		f.write('\n %.3f / %.3f ' %(count_iter, irreg ) )
		f.write('\n final correct %.3f' %(final_thresh))
	








#				with open('model'+'_'+str(lrate)+'_'+str(hd_sz)+str(fold)+'.pt', 'wb') as f:
#					torch.save(model, f)
#				
#			with open('all_results_f', 'a') as f:
#				f.write(str(lrate)+'_'+str(hd_sz)+'\n')
#				f.write('Train -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_tr).mean(),np.array(mae_tr).mean(), np.array(mzo_tr).mean() ) )
#				f.write('Test -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_te).mean(),np.array(mae_te).mean(), np.array(mzo_te).mean() ) )
#				f.write('Train _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_tr).std(),np.array(mae_tr).std(), np.array(mzo_tr).std() ) )
#				f.write('Test _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_te).std(),np.array(mae_te).std(), np.array(mzo_te).std() ) )
#			
#			
#	
#			s = 'Total Training Time: %.2f min' % ((time.time() - start_time)/60)
#			print(s)
#			with open(LOGFILE+'_'+str(lrate)+'_'+str(hd_sz), 'a') as f:
#			    f.write('%s\n' % s)
#				
#						
#				
	
