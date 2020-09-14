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

parser = argparse.ArgumentParser()
parser.add_argument("-h_l",dest='hidden_layer' ,help="hidden layer size", type=int, nargs='+')
parser.add_argument("-lr",dest='l_rate' ,help="learning rate", type=float, nargs='+')
parser.add_argument("-n", dest="noise_model", help="noise rate", type=int)
args = parser.parse_args()
ind = args.noise_model


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#pre - processing 
df = pd.read_csv('../../../abalone.data', header=None)

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
LOGFILE = 'log'








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
#        for j in range(num_classes):
#                if i!=j:
#                        if ind==0:
#                                eta_n[i][j] = 0
#                        if ind==1:
#                                eta_n[i][j] = 0.05/abs(i-j)
#                        if ind==2:
#                                eta_n[i][j] = 0.1/abs(i-j)
#                        if ind==3:
#                                eta_n[i][j] = 0.15/abs(i-j)
#                        if ind==4:
#                                eta_n[i][j] = np.exp(-3*abs(i-j))
#                        if ind==5:
#                                eta_n[i][j] = np.exp(-2.2*abs(i-j))
#                        if ind==6:
#                                eta_n[i][j] = np.exp(-1.8*abs(i-j))
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
mse_tr = []
mae_tr = []
mzo_tr = []
mse_te = []
mae_te = []
mzo_te = []


np.random.seed(1)
df[label] = df[label].map(noii)

#df = df.head(100)

dat = CustomDataset(df, label, num_classes) 
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

for lrate in args.l_rate:
	for hd_sz in args.hidden_layer:
		
		mse_tr = []
		mae_tr = []
		mzo_tr = []
		mse_te = []
		mae_te = []
		mzo_te = []
		fold = 0
		kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
		for train_index, test_index in kfold.split(df.values, df[[label]].values):
			fold += 1
			train_sampler = SubsetRandomSampler(train_index)
			test_sampler = SubsetRandomSampler(test_index)
			dat_train = torch.utils.data.DataLoader(dataset=dat, batch_size=batch_size, sampler=train_sampler)
			dat_test = torch.utils.data.DataLoader(dataset=dat, batch_size=batch_size, sampler=test_sampler)
			torch.manual_seed(1)
			model = NeuralNet(input_size=input_size, hidden_size1=hd_sz, num_classes=num_classes)
			#model = model.float()
			model.to(device, dtype=torch.float64)
			start_time = time.time()
			optimizer = torch.optim.AdamW(model.parameters(), lr=lrate)
			
			for epoch in range(num_epochs):
				model.train()
				running_loss = 0.0
				total_rows = 0
				if epoch%100==0 & epoch > 0:
				    lrate/=2
			  
				optimizer = torch.optim.AdamW(model.parameters(), lr=lrate)
				for batch_idx, (features, l) in enumerate(dat_train):
					features = features.to(device)
					l = l.to(device)
					logits, probas = model(features)
					#logits.to(torch.cuda.DoubleTensor)
					cost = l_hat(logits, l, eta_inv, num_classes)
					optimizer.zero_grad()
					cost.backward()
					optimizer.step()
					total_rows += l.size(0)
					running_loss += cost.item()*l.size(0)
				s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
				ls = 'Epochs=%f, Total Loss=%.3f, Loss(per_example)=%.3f, Examples=%f' % (epoch, running_loss,running_loss/total_rows,total_rows)
				print('%s \n%s\n' % (ls,s))
				with open(LOGFILE+'_'+str(lrate)+'_'+str(hd_sz), 'a') as f:
					f.write('%s \n%s\n' % (ls,s))

				model = model.to(torch.float64)
				model.eval()
				with torch.set_grad_enabled(False):  # save memory during inference
					#train_mae, train_mse, train_mzo, ex1 = compute_mae_and_mse(model, dat_train,device=device)
					test_mae, test_mse, test_mzo, ex2 = compute_mae_and_mse(model, dat_test,device=device)
					#s = 'Train - MAE/RMSE/MZO/EX: %.2f/%.2f/%.2f/%.2f \n ' % (train_mae, torch.sqrt(train_mse), train_mzo, ex1)
					#print(s)
					with open(LOGFILE+'_'+str(lrate)+'_'+str(hd_sz), 'a') as f:
						f.write('%f' % epoch)
					#	f.write('\n%s\n' % s)
					
					s = 'Test - MAE/RMSE/MZO/EX:  %.2f/%.2f/%.2f/%.2f \n' % (test_mae, torch.sqrt(test_mse), test_mzo, ex2)
					print(s)
					with open(LOGFILE+'_'+str(lrate)+'_'+str(hd_sz), 'a') as f:
						f.write('%s\n' % s)
								    		    
			
			model = model.to(torch.float64)
			model.eval()
			with torch.set_grad_enabled(False):  # save memory during inference
				train_mae, train_mse, train_mzo, ex1 = compute_mae_and_mse(model, dat_train,device=device)
				test_mae, test_mse, test_mzo, ex2 = compute_mae_and_mse(model, dat_test,device=device)
				s = 'Train - MAE/RMSE/MZO/EX: %.2f/%.2f/%.2f/%.2f \n ' % (train_mae, torch.sqrt(train_mse), train_mzo, ex1)
				print(s)
				with open(LOGFILE+'_'+str(lrate)+'_'+str(hd_sz), 'a') as f:
					f.write('%f\n%s\n' %(fold, s))
				with open('all_results', 'a') as f:
					f.write(LOGFILE+'_'+str(lrate)+'_'+str(hd_sz)+'\n%s\n' % s)
				
				s = 'Test - MAE/RMSE/MZO/EX:  %.2f/%.2f/%.2f/%.2f \n' % (test_mae, torch.sqrt(test_mse), test_mzo, ex2)
				print(s)
				with open(LOGFILE+'_'+str(lrate)+'_'+str(hd_sz), 'a') as f:
				    f.write('%s\n' % s)
				with open('all_results', 'a') as f:
				    f.write('%s\n' % s)
				
							

			mse_tr.append(np.array(torch.sqrt(train_mse).cpu()))
			mae_tr.append(np.array(train_mae.cpu()))
			mzo_tr.append(np.array(train_mzo.cpu()))
			mse_te.append(np.array(torch.sqrt(test_mse).cpu()))
			mae_te.append(np.array(test_mae.cpu()))
			mzo_te.append(np.array(test_mzo.cpu()))




	
			with open('model'+'_'+str(lrate)+'_'+str(hd_sz)+str(fold)+'.pt', 'wb') as f:
				torch.save(model, f)
			
		with open('all_results_f', 'a') as f:
			f.write(str(lrate)+'_'+str(hd_sz)+'\n')
			f.write('Train -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_tr).mean(),np.array(mae_tr).mean(), np.array(mzo_tr).mean() ) )
			f.write('Test -- MSE / MAE / MZO : %.3f / %.3f / %.3f\n' %(np.array(mse_te).mean(),np.array(mae_te).mean(), np.array(mzo_te).mean() ) )
			f.write('Train _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_tr).std(),np.array(mae_tr).std(), np.array(mzo_tr).std() ) )
			f.write('Test _ std -- MSE / MAE / MZO : %.4f / %.4f / %.4f\n' %(np.array(mse_te).std(),np.array(mae_te).std(), np.array(mzo_te).std() ) )
		
		

		s = 'Total Training Time: %.2f min' % ((time.time() - start_time)/60)
		print(s)
		with open(LOGFILE+'_'+str(lrate)+'_'+str(hd_sz), 'a') as f:
		    f.write('%s\n' % s)
			
					
			
# Train - MAE/RMSE/EX: | Train: 0.44/0.70/3342.00 
# Test - MAE/RMSE/EX: | Train: 0.44/0.70/835.00 :w

# Total Training Time: 9.03 min
# 
# 

# In[154]:

