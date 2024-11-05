# -*- coding: utf-8 -*-
import pickle
import time
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d
from Evaluation import test_epoch,SignalDataset,cal_statistic
from TSDNet import TSDNet_B
from AHI_regressionNet import AHINet
import warnings
warnings.filterwarnings('ignore',category=UserWarning,module='matplotlib.font_manager')

batch_size = 512*4
class_num = 2
SIG_LEN = 256
feature_attn_len=256


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def resample_array_2d(array, target_length):
    original_length = array.shape[0]
    x_old = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, target_length)
    
    f0 = interp1d(x_old, array[:, 0], kind='linear')
    f1 = interp1d(x_old, array[:, 1], kind='linear')

    resampled_col0 = f0(x_new)
    resampled_col1 = f1(x_new)

    resampled_array = np.stack((resampled_col0, resampled_col1), axis=-1)
    
    return resampled_array


def extract_and_convert(path, column_name):
    df = pd.read_csv(path)

    df[column_name] = df[column_name].astype(str)

    mask = df[column_name].str.contains('M', na=False)
    filtered_df = df[~mask]
    

    removed_indices = df[mask].index
    

    try:
        extracted_values = filtered_df[column_name].astype(float)
    except ValueError as e:
        print(f"Error converting column {column_name} to float: {e}")
        extracted_values = filtered_df[column_name].str.replace(',', '').astype(float)
    
    return extracted_values.values, removed_indices


test_model_name = './weight/tsdnet.chkpt'#SpO2

model = TSDNet_B()
chkpoint = torch.load(test_model_name, map_location='cuda')
model.load_state_dict(chkpoint)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#cfs
folder_path = r'./test_data/'
data_pattern = 'data_all_270s_cfs-visit5-*.edf.txt'
label_pattern = 'label_all_270s_cfs-visit5-*.edf.txt'
data_list = glob.glob(folder_path + data_pattern)
label_list = glob.glob(folder_path + label_pattern)

All_pred = []
True_AHI = []
Pred_AHI = []
patient_num = 60
for i in tqdm(range(patient_num)):
    data_file_path = data_list[i]
    patient_id = data_file_path.split("data_all_270s_")[1].rsplit(".edf.txt")[0]
    print(f"Patient ID: {patient_id}")
    with open(data_file_path, 'rb') as file:
        Test_ecg_data_s_pick = pickle.load(file)
        Test_ecg_data_s_pick = Test_ecg_data_s_pick[:,2,:][:,np.newaxis,:]
        mean = np.mean(Test_ecg_data_s_pick, axis=2, keepdims=True)
        std = np.std(Test_ecg_data_s_pick, axis=2, keepdims=True)
        Test_ecg_data_s_pick = (Test_ecg_data_s_pick - mean) / (std + 1e-8)  # 避免除以零
        Test_ecg_data_s_pick = Test_ecg_data_s_pick.astype('float32')

    label_file_path = label_list[i]
    with open(label_file_path, 'rb') as file:
        Test_label_s_pick = pickle.load(file)       

    Test_label_s_pick[Test_label_s_pick == 2] = 1
    
    count = len(np.argwhere(Test_label_s_pick==1))
    ahi = count*120/len(Test_label_s_pick)
    test_data = SignalDataset(Test_ecg_data_s_pick,Test_label_s_pick)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             num_workers=0,
                             shuffle=False)


    start = time.time()
    cm,all_pred = test_epoch(test_loader, device, model, test_data.__len__(),0.9, class_num)
    All_pred.append(resample_array_2d(all_pred,1024))
    acc, sen, spe, ppv, F1 = cal_statistic(cm)


All_pred = np.array(All_pred)


path = r'./cfs.csv'
column_name = 'nsrr_ahi_hp3r_aasm15'

Nsrr_ahi,removed_indices = extract_and_convert(path, column_name)
Nsrr_ahi = Nsrr_ahi[:patient_num]
All_pred = np.delete(All_pred, removed_indices, axis=0)
All_pred = All_pred.transpose(0,2,1)
print(All_pred.shape)
print(Nsrr_ahi.shape)


dataset = MyDataset(All_pred, Nsrr_ahi)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

model = AHINet()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
opt_path = r'./weight/AHI_regression.pth'
model.load_state_dict(torch.load(opt_path))
model.eval()
true_values = []
pred_values = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        true_values.extend(targets.cpu().numpy())
        pred_values.extend(outputs.cpu().numpy())


true_values = np.array(true_values).flatten()
pred_values = np.array(pred_values).flatten()
    
plt.figure(figsize=(16,2))
plt.plot(true_values,label='True AHI')
plt.plot(pred_values,label='Pred AHI')
plt.legend()
plt.tight_layout()
plt.show()

risk_values = np.mean(All_pred[:,1,:],axis=-1)

for idx, (true, pred, risk) in enumerate(zip(true_values, pred_values, risk_values)):
    print(f'Index: {idx}, True: {true:.4f}, Predicted: {pred:.4f}, Risk: {risk:.4f}')
    

df = pd.DataFrame({    
    'true_values': true_values,
    'pred_values': pred_values,
    'pred_risk':risk_values
})


df.to_csv(r'./pred_results.csv', index=False)