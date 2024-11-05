from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp


class SignalDataset(Dataset):
    def __init__(self, raw_data, raw_label):

        self._signal = torch.FloatTensor(raw_data)
        self._label = torch.LongTensor(raw_label)
    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._label)

    @property
    def sig_len(self):
        return self._signal.shape[-1]

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self._signal[idx], self._label[idx]


def test_epoch(valid_loader, device, model, total_num, threshold, class_num):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    cnt_per_class = np.zeros(class_num)
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=0.5, desc='- (Validation)  ', leave=False):
            sig, label, = map(lambda x: x.to(device), batch)
            pred = model(sig)
            all_labels.extend(label.cpu().numpy())

            prob_class_2 = pred[:, 1]
            prob_class_1 = pred[:, 0]

            condition = (prob_class_2 >= threshold) & (prob_class_2 > prob_class_1)

            predictions = torch.where(condition, torch.tensor(1, device=device), torch.tensor(0, device=device))

            all_res.extend(predictions.cpu().numpy())

            all_pred.extend(pred.cpu().numpy())
            loss, n_correct, cnt = cal_loss(pred, label.squeeze(), device, class_num)

            total_loss += loss.item()
            total_correct += n_correct
            cnt_per_class += cnt

    np.savetxt('all_pred.txt',all_pred)
    np.savetxt('all_label.txt', all_labels)
    all_pred = np.array(all_pred)
    
    cm = confusion_matrix(all_labels, all_res)
    print(cm)
    if len(cm)!=1 and np.sum(cm[-(len(cm)-1),:]!=0):
        FPR, TPR, AUC,_,_,_ = plot_roc(all_labels,all_pred,class_num)
    acc, sen, spe, ppv, F1 = cal_statistic(cm)
    print('acc is : {acc}'.format(acc=acc))
    print('sen is : {sen}'.format(sen=sen))
    print('spe is : {spe}'.format(spe=spe))
    print('ppv is : {ppv}'.format(ppv=ppv))
    print('F1 is : {F1}'.format(F1=F1))
    test_acc = total_correct / total_num
    print('test_acc is : {test_acc}'.format(test_acc=test_acc))

    return cm,all_pred


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.average = average

        if alpha is None:
            self.alpha = torch.ones(class_num)
        else:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                self.alpha = torch.Tensor(alpha)

    def forward(self, inputs, targets, device):
        N, C = inputs.size()
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.new_zeros(N, C)
        ids = targets.view(-1, 1) 
        class_mask.scatter_(1, ids, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(device)
        alpha = self.alpha[ids.view(-1)].view(-1, 1)

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()

        batch_loss = -alpha * torch.pow((1 - probs), self.gamma) * log_p
        
        if self.average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def cal_loss(pred, label, device, class_num):
    cnt_per_class = np.zeros(class_num)
    alpha = [0.5,0.5]
    FL = FocalLoss(class_num, alpha=alpha)
    loss = FL(pred, label, device)
    pred = pred.max(1)[1]
    n_correct = pred.eq(label).sum().item()
    cnt_per_class = [cnt_per_class[j] + pred.eq(j).sum().item() for j in range(class_num)]
    return loss, n_correct, cnt_per_class


def cal_statistic(cm):
    tp = np.diagonal(cm)
    gt_num = np.sum(cm, axis=1)
    pre_num = np.sum(cm, axis=0)
    fp = pre_num - tp
    num0 = np.sum(gt_num)
    num = np.repeat(num0, gt_num.shape[0])
    gt_num0 = num - gt_num
    tn = gt_num0 -fp
    sen = tp.astype(np.float32) / gt_num
    spe = tn.astype(np.float32) / gt_num0
    ppv = tp.astype(np.float32) / pre_num
    F1 = 2 * (sen * ppv) / (sen + ppv)
    acc = np.sum(tp).astype(np.float32) / num0

    sen[np.isnan(sen)] = 0
    spe[np.isnan(spe)] = 0
    ppv[np.isnan(ppv)] = 0
    F1[np.isnan(F1)] = 0
    return acc, sen, spe, ppv, F1


def plot_roc(all_labels, all_pred, class_num):
    enc = OneHotEncoder()
    all_labels = np.array(all_labels)

    label_h =enc.fit_transform(all_labels)
    label_h = label_h.toarray()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(class_num):
        fpr[i], tpr[i], _ = roc_curve(label_h[:, i], all_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_num)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(class_num):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= class_num
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fpr["micro"], tpr["micro"], _ = roc_curve(label_h.ravel(), all_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fig, ax = plt.subplots(figsize=(4,3))
    yminorLocator = MultipleLocator(0.1)
    xminorLocator = MultipleLocator(0.1)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro roc (AUC = {0:0.3f})'
                   ''.format(roc_auc["micro"]),
             color='#9C5BCD', linewidth=3)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro roc (AUC = {0:0.3f})'
                   ''.format(roc_auc["macro"]),
              linestyle='-', color='#87B5B2',linewidth=3)

    plt.plot([0, 1], [0, 1], lw=1,color='black',linestyle='-.')
    plt.xlim([-0.01, 1.0])
    plt.ylim([-0.01, 1.01])
    plt.grid(which='both', color='grey', ls='--', lw=0.5)
    plt.xlabel('1-Specificity',fontsize=12)
    plt.ylabel('Sensitivity',fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower right',fontsize=12,frameon=True)

    plt.tight_layout()
    plt.show()
    return fpr["micro"],tpr["micro"],roc_auc["micro"],fpr["macro"],tpr["macro"],roc_auc["macro"]