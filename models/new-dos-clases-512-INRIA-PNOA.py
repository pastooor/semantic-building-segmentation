import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
#from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
import getopt, sys
from tqdm.notebook import tqdm
from tqdm import tqdm

# from torchsummary import summary
import segmentation_models_pytorch as smp
import seaborn as sns
import xlsxwriter

#nuevos
import datetime
import random

import torchvision.utils as vutils

from torch import optim
from torch.backends import cudnn



os.environ["CUDA_VISIBLE_DEVICES"]="0"  
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TRAIN_IMAGE_PATH = 'G:\\jorge\\Datasets\\INRIA-PNOA\\train\\images\\'
TRAIN_MASK_PATH = 'G:\\jorge\\Datasets\\INRIA-PNOA\\train\\masks\\'
TEST_IMAGE_PATH = 'G:\\jorge\\Datasets\\INRIA-PNOA\\test\\images\\'
TEST_MASK_PATH = 'G:\\jorge\\Datasets\\INRIA-PNOA\\test\\masks\\'
VAL_IMAGE_PATH = 'G:\\jorge\\Datasets\\INRIA-PNOA\\val\\images\\'
VAL_MASK_PATH = 'G:\\jorge\\Datasets\\INRIA-PNOA\\val\\masks\\'

# ESTABLECER N?? de CLASES
n_classes = 2 

# ESTABLECER N?? EPOCHs
epochs = 250
lr_patience = 5
print_freq = 100
val_img_sample_rate = 0.1 #1
# N iteraciones sin mejora pertimitidas (loss, miou)
Max_NON_Improve_Iterations= 15
Cont_Improve_Iterations = 0

# batch_sizes validations
batch_sizes_val = 1

#img size
img_size = 512

# Concat models names
ModelNamesArray = ''
args = {
    'lr': 1e-10,
    'val_save_to_img_file': False
}
args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iou': 0, 'fwavacc': 0}

# scheduler name for reports and model filenames (ReduceLROnPlateau or OneCycleLR)
sche_name = 'ReduceLROnPlateau'



def create_df(IMAGE_PATH):
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            name.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

trdf = create_df(TRAIN_IMAGE_PATH)
ttdf = create_df(TEST_IMAGE_PATH)
vdf = create_df(VAL_IMAGE_PATH)
X_train =  trdf['id'].values.tolist() #train_test_split(trdf['id'].values, test_size=0.9999)
X_test = ttdf['id'].values.tolist() #train_test_split(ttdf['id'].values, test_size=0.999)
X_val = vdf['id'].values.tolist() #train_test_split(vdf['id'].values, test_size=0.999)



print('Train Size   : ', len(X_train))
print('Val Size     : ', len(X_val))
print('Test Size    : ', len(X_test))



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, n=1):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count
        
    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

class SROADEXDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, mean, std, transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):

        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            #img = Image.fromarray(aug['image']) # esto estaba antes y funcionaba
            img = aug['image']
            mask = aug['mask']
        
        if self.transform is None:
            img = Image.fromarray(img)
       
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img.astype( np.float32) / 255.)
        mask = torch.from_numpy(mask).long()

        return img, mask, self.X[idx]
    
    def tiles(self, img, mask):

        img_patches = img.unfold(1, 256, 256).unfold(2, 768, 768) 
        img_patches  = img_patches.contiguous().view(3,-1, 512, 768) 
        img_patches = img_patches.permute(1,0,2,3)
        
        mask_patches = mask.unfold(0, 256, 256).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)
        
        return img_patches, mask_patches

#mean = [0.533, 0.536, 0.473]
#std  = [0.184, 0.170, 0.166]   
mean = [0.372, 0.362, 0.344]
std  = [0.307, 0.294, 0.281]    

t_train = A.Compose([A.Resize(512, 512, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(), A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),        A.GaussNoise()])

t_val = A.Compose([A.Resize(512, 512, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                   A.GridDistortion(p=0.2)])

                   
#datasets
train_set = SROADEXDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, X_train, mean, std, t_train, patch=False)
val_set = SROADEXDataset(VAL_IMAGE_PATH, VAL_MASK_PATH, X_val, mean, std, t_val, patch=False)



def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=1e-10, n_classes= n_classes):
    with torch.no_grad():
#        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)

    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluate_old(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iou, fwavacc

def evaluate_1(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):      
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    return hist

def evaluate_all(hist):
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iou = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iou, fwavacc

def train_old(train_loader, model, criterion, optimizer, epoch, p_freq):
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)

    for i, data in enumerate(train_loader):
        inputs, labels = data
        assert inputs.size()[2:] == labels.size()[1:]
        N = inputs.size(0)
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == n_classes

        loss = criterion(outputs, labels) / N
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data, N)

        curr_iter += 1

        if (i + 1) % p_freq == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg))

def train(train_loader, model, criterion, optimizer, epoch, p_freq):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target, fimg) in enumerate(stream, start=1):
        images = Variable(images).cuda()
        target = Variable(target).cuda()
        output = model(images) #.squeeze(1)
        loss = criterion(output, target)
        metric_monitor.update("Loss", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )

                
def validate(val_loader, model, model_n, criterion, optimizer, epoch, train_args, fit_time):
    histsum = np.zeros((n_classes, n_classes))
    metric_monitor = MetricMonitor()
    improve = 0
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target, fimg) in enumerate(stream, start=1):
            images = Variable(images).cuda()
            target = Variable(target).cuda()
            output = model(images)         
            loss = criterion(output, target)
            metric_monitor.update("Loss", loss.item())
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )
            histsum += evaluate_1(output.data.max(1)[1].squeeze_(1).cpu().numpy(),
            			   target.data.cpu().numpy(), n_classes)            
    acc, acc_cls, mean_iou, fwavacc = evaluate_all(histsum)

    if mean_iou > train_args['best_record']['mean_iou']:
        train_args['best_record']['val_loss'] = metric_monitor.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iou'] = mean_iou
        train_args['best_record']['fwavacc'] = fwavacc
        improve= 1

        print('-----------------------------------------------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iou %.5f], [fwavacc %.5f], [improve_iter %d]' % (
        epoch, metric_monitor.avg, acc, acc_cls, mean_iou, fwavacc, improve))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iou %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iou'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch']))
    print('Epoch time: {:.2f} m' .format((time.time()- fit_time)/60))
    print('-----------------------------------------------------------------------------------------------------------')

    history = {'val_loss': metric_monitor.avg,
               'mean_iou': mean_iou,
               'acc' :acc, 'acc_cls':acc_cls, 'improve': improve,
               'fwavacc': fwavacc, 'time': (time.time()- fit_time)/60}
#    model.train()
    return metric_monitor.avg, history

def validate_old(val_loader, model, model_n, criterion, optimizer, epoch, train_args, fit_time): #, restore, visualize):
    model.eval()
    improve = 0
    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []
    with torch.no_grad():
        for vi, data in enumerate(val_loader):       
            inputs, gts = data
            N = inputs.size(0)
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()

            outputs = model(inputs)
            predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()

            val_loss.update(criterion(outputs, gts).data / N, N)

            for i in inputs:
                if random.random() > val_img_sample_rate:
                    inputs_all.append(None)
                else:
                    inputs_all.append(i.data.cuda())
            gts_all.append(gts.data.cpu().numpy())
            predictions_all.append(predictions)

    gts_all = np.concatenate(gts_all)
    predictions_all = np.concatenate(predictions_all)

    acc, acc_cls, mean_iou, fwavacc = evaluate(predictions_all, gts_all, n_classes)

    if mean_iou > train_args['best_record']['mean_iou']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iou'] = mean_iou
        train_args['best_record']['fwavacc'] = fwavacc

        improve= 1

        print('-----------------------------------------------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iou %.5f], [fwavacc %.5f], [improve_iter %d]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iou, fwavacc, improve))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iou %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iou'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch']))
    print('Epoch time: {:.2f} m' .format((time.time()- fit_time)/60))
    print('-----------------------------------------------------------------------------------------------------------')

    history = {'val_loss': val_loss.avg.cpu().numpy(),
               'mean_iou': mean_iou,
               'acc' :acc, 'acc_cls':acc_cls, 'improve': improve,
               'fwavacc': fwavacc, 'time': (time.time()- fit_time)/60}
    model.train()
    return val_loss.avg, history



# FUNCIONES CREADAS PARA GUARDAR HISTORIA ENTRENAMIENTO Y MATRIZ DE CONFUSION EN UN SOLO ARCHIVO EXCELL

def save_confusion_matrix(matrix, model_name, file_name):
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet(model_name)
    for j in range(0, matrix.shape[1]):
        worksheet.write(0, j+1, j+1)   
    row = 1
    for i in range(0, matrix.shape[0]):
        worksheet.write(row, 0, row)
        for j in range(0, matrix.shape[1]):
            worksheet.write(row, j+1, matrix[j][i])       
        row += 1        
    worksheet.write('G1','P1')
    worksheet.write('H1','P2')

    worksheet.write('K1','R1')
    worksheet.write('L1','R2')
  
    worksheet.write('G5','F1-1')
    worksheet.write('H5','F1-2')

    worksheet.write('L5','MIoU')
    worksheet.write('M5','Acc testing')
    worksheet.write('G8','IoU-1')
    worksheet.write('H8','IoU-2')
 

    
    worksheet.write_formula('G2', '=B2/SUM(B2:E2)')
    worksheet.write_formula('H2', '=C3/SUM(B3:E3)')


    worksheet.write_formula('K2', '=B2/SUM(B2:B5)')
    worksheet.write_formula('L2', '=C3/SUM(C2:C5)')


    worksheet.write_formula('G6', '=2*G2*K2 /(G2+K2)')
    worksheet.write_formula('H6', '=2*H2*L2 /(H2+L2)')


    worksheet.write_formula('G9', '=B2/(SUM(B2:B5) + SUM(C2:E2))')
    worksheet.write_formula('H9', '=C3 /(SUM(B3:E3)+ C2+C4+C5)')

    
    worksheet.write_formula('L6', '=SUM(G9:H9)/2')
    worksheet.write_formula('M6', '=(B2+C3) /SUM(B2:C3)')
  

    workbook.close()

def save_scores_imgs(scores, model_name, file_name, score_type):

    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet(model_name)
    worksheet.write(0, 0, score_type)
    worksheet.write(0, 1, 'Img file')
    for j in range(0, len(scores)):
        worksheet.write(j+1, 0, scores[j]['val'])   
        worksheet.write(j+1, 1, scores[j]['img'])   

    workbook.close()
   
def save_loss_score_acc(history, model_name, file_name):
    workbook = xlsxwriter.Workbook(file_name)
    i = 0
#    for i in range(0, len(model_name)):
    worksheet = workbook.add_worksheet(model_name)

    worksheet.write(0, 0, 'Loss Validation')
    worksheet.write(0, 1, 'MIoU Validation')
    worksheet.write(0, 2, 'Accuracy Validation')
    worksheet.write(0, 3, 'Accuracy Classification')    
    worksheet.write(0, 4, 'fwavacc') 
    worksheet.write(0, 5, 'Tiempo') 
    row = 1

    for j in range(0, len(history[i]['val_loss'])):
        worksheet.write(j+1, 0, history[i]['val_loss'][j])   
    if j > row: row = j 

    for j in range(0, len(history[i]['mean_iou'])):
        worksheet.write(j+1, 1, history[i]['mean_iou'][j])   
    if j > row: row = j

    for j in range(0, len(history[i]['acc'])):
        worksheet.write(j+1, 2, history[i]['acc'][j])   
    if j > row: row = j

    for j in range(0, len(history[i]['acc_cls'])):
        worksheet.write(j+1, 3, history[i]['acc_cls'][j])   
    if j > row: row = j

    for j in range(0, len(history[i]['fwavacc'])):
        worksheet.write(j+1, 4, history[i]['fwavacc'][j])   
    if j > row: row = j
        
    for j in range(0, len(history[i]['duration'])):
        worksheet.write(j+1, 5, history[i]['duration'][j])   
    workbook.close()
        
def plot_loss(history, model_name, itera):
    plt.figure(figsize = (5, 5), dpi = 125)
    plt.plot(history['val_loss'], label='val_loss', marker='o')
    plt.title('Loss per epoch'); plt.ylabel('loss');
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(f'{model_name}_{str(itera)}-Loss.jpg', bbox_inches = 'tight')
#    plt.show()
    
def plot_score(history, model_name, itera):
    plt.figure(figsize = (5, 5), dpi = 125)
    plt.plot(history['mean_iou'], label='mean_iou',  marker='.')
    plt.title('Score per epoch'); plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(f'{model_name}_{str(itera)}-Mean IoU.jpg',bbox_inches = 'tight')
#    plt.show()
    
def plot_acc(history, model_name, itera):
    plt.figure(figsize = (5, 5), dpi = 125)
    plt.plot(history['acc'], label='val_accuracy',  marker='p')
    plt.title('Accuracy per epoch'); plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(f'{model_name}_{str(itera)}-Accuracy.jpg', bbox_inches = 'tight')
#    plt.show()
    
class SROADEXTestDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std
              
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
        
        if self.transform is None:
            img = Image.fromarray(img)
       
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img.astype( np.float32) / 255.)
        mask = torch.from_numpy(mask).long()
                     
        return img, mask, self.X[idx]

    
def pixel_acc(model, test_set):
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask, fimg = test_set[i]
        pred_mask, acc = predict_image_mask_pixel(model, img, mask)
        acc_ = {'val': acc, 'img': fimg}
        accuracy.append(acc_)
    return accuracy

def miou_score(model, test_set):
    score_iou = []
    for i in tqdm(range(len(test_set))):
        img, mask, fimg = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_ = {'val': score, 'img': fimg}
        score_iou.append(score_)
    return score_iou


def predict_image_mask_miou(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
#    model.eval()
#    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
#    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score

def predict_image_mask_pixel(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
#    model.eval()
#    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
#    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc

def train_model(model, model_n, batch_size_train, epochs, Cont_Improve_Iterations, iteracion):
    val_losses = []
    mean_iou = []; acc = []
    acc_cls = []; fwavacc = []; duration = []
    lrs = []
    non_improve = 0

    max_lr = 1e-3
    weight_decay = 1e-4

    #dataloader

    visualize = T.ToTensor()
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_sizes_val, shuffle=True)  

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
#    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.95, weight_decay=weight_decay)
#    optimizer = torch.optim.NAdam(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    
    #train_losses.append y lo mismo con el resto de par??metros de los entrenos
   
    history = []
    if torch.cuda.is_available():
        model.cuda()
    if sche_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience= lr_patience, factor=0.1, cooldown= 0, min_lr=1e-5) #, verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(1, epochs +1): ## iteracion inicial modificada para restaurar punto de ruptura
        fit_time = time.time()
        train(train_loader, model, criterion, optimizer, epoch, print_freq)
        val_loss, history_ = validate(val_loader, model, model_n, criterion, optimizer, epoch, args, fit_time)#, restore_transform, visualize)
        val_losses.append(history_['val_loss'])
        mean_iou.append(history_['mean_iou'])
        acc.append(history_['acc'])
        acc_cls.append(history_['acc_cls'])
        fwavacc.append(history_['fwavacc'])
        duration.append(history_['time'])        
        scheduler.step(val_loss)
        Cont_Improve_Iterations += history_['improve']
        if history_['improve'] == 0:
            non_improve += 1
            if non_improve >= Max_NON_Improve_Iterations:
                print('saving model and exit because not improve during %d epochs...' % (non_improve))
                break            
        if epoch > 4 and history_['improve'] == 1:
            non_improve = 0
            epoch_name = '%s_best_model_%d_itera' % (model_n, iteracion)
            print('saving model...')
            if os.path.exists(os.path.join(epoch_name + '.pt')):
                os.remove(os.path.join(epoch_name + '.pt'))
            torch.save(model, os.path.join(epoch_name + '.pt'))


    history = {'val_loss': val_losses,
               'mean_iou': mean_iou,
               'acc' :acc, 'acc_cls':acc_cls,
               'fwavacc': fwavacc, 'duration': duration}
    return history


# Leer par??metros de la llamada
argumentList = sys.argv[1:] 
# Options 
options = "hn:m:c:"
# Long options 
long_options = ["Help", "Ordinal=","itera=","checkpoint="] 

index = -1
itera = -1

checkpoint_path = None

try: 
    # Parsing argument 
    arguments, values = getopt.getopt(argumentList, options, long_options) 
      
    # checking each argument 
    for currentArgument, currentValue in arguments: 
  
        if currentArgument in ("-h", "--Help"): 
            print ("Diplaying Help")
            print ("-n Red; -m iteracion")    
        elif currentArgument in ("-n", "--Ordinal"): 
            print ("Ordinal (% s)" % (currentValue))
            index = int(currentValue) -1
        elif currentArgument in ("-m", "--itera"):
            print ("iteracion (% s)" % (currentValue))
            itera = int(currentValue)
        elif currentArgument in ("-c", "--checkpoint"):
            print(f"Usar checkpoint: {currentValue}")
            checkpoint_path = currentValue                          
except getopt.error as err: 
    # output error, and return with an error code 
    print (str(err))



if index < 0: 
    sys.exit("No hay parametro Ordinal (n)")



t_test = A.Resize(512, 512, interpolation=cv2.INTER_NEAREST)

test_set = SROADEXTestDataset(TEST_IMAGE_PATH, TEST_MASK_PATH, X_test, transform= t_test)


# DeepLabV3Plus_Resnext101_32x8d
ENCODER = 'resnext101_32x8d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = n_classes

DeepLabV3Plus_Resnext101_32x8d = smp.DeepLabV3Plus(
    encoder_name = ENCODER,
    encoder_weights = ENCODER_WEIGHTS,
    classes = CLASSES)


# UNet-Resnet34
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = n_classes

UNet_Resnet34 = smp.Unet(
    encoder_name = ENCODER,
    encoder_weights = ENCODER_WEIGHTS,
    classes = CLASSES)


# PAN_Mobilenet_v2
ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = n_classes


PAN_Mobilenet_v2 = smp.PAN(
    encoder_name = ENCODER,
    encoder_weights = ENCODER_WEIGHTS,
    classes = CLASSES) 


# FPN-Resnet101
ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = n_classes


FPN_Resnet101 = smp.FPN(
    encoder_name = ENCODER,
    encoder_weights = ENCODER_WEIGHTS,
    classes = CLASSES)
   
# UPerNet-Convnext_base
ENCODER = 'tu-convnext_base'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = n_classes

UPerNet_Convnext_base = smp.UPerNet(
    encoder_name = ENCODER,
    encoder_weights = ENCODER_WEIGHTS,
    classes = CLASSES)


model_names = ['DeepLabV3Plus_Resnext101_32x8d', 'UNet_Resnet34', 'PAN_Mobilenet_v2', 'FPN_Resnet101', 'UPerNet_Convnext_base']
models = [ DeepLabV3Plus_Resnext101_32x8d, UNet_Resnet34, PAN_Mobilenet_v2, FPN_Resnet101, UPerNet_Convnext_base]
batch_sizes =  [10,10,10,10,10]

history = []

# COMIENZA A PROCESAR LOS MODELOS


print("Añade Modelo: ", model_names[index])
model = models[index]

#Si se pasó checkpoint, cargar pesos aquí
if checkpoint_path:
    print(f"Cargando checkpoint desde {checkpoint_path} …")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ck, torch.nn.Module):
        model = ck.to(device)
        print("Modelo completo cargado desde checkpoint")
    else:
        sd = ck.get('state_dict', ck)
        model.load_state_dict(sd)
        model = model.to(device)
        print("State_dict cargado en model")


# modelo rescatado
#t_model = torch.load(f'./unet-incepv2-64-32-16-8_23-ReduceLROnPlateau_model.pt')

Cont_Improve_Iterations = 0
ModelNamesArray = model_names[index]
#history.append(train_model(t_model, model_names[index], batch_sizes[index], epochs, Cont_Improve_Iterations, itera))  #models[index]  modificado
history.append(train_model(model,
                          model_names[index],
                          batch_sizes[index],
                          epochs,
                          Cont_Improve_Iterations,
                          itera))
torch.save(models[index], f'{model_names[index]}_{str(itera)}-{sche_name}_model.pt')
torch.cuda.empty_cache()

    
histarr = np.array(history)


print(f"Plot Loss, Score, Accuracy for Model = {model_names[index]}")
plot_loss(history[0], model_names[index], itera)
plot_score(history[0], model_names[index], itera)
plot_acc(history[0], model_names[index], itera)

plt.figure(figsize = (5, 5), dpi = 125)

plt.plot(histarr[0]['val_loss'], marker='o')

plt.title('Score per epoch'); plt.ylabel('Validation Loss')
plt.xlabel('epoch')
plt.legend(labels = model_names), plt.grid()
plt.savefig(ModelNamesArray + '_'+ str(itera) + '-' + sche_name+ '-Validation_Loss.jpg', bbox_inches = 'tight')


plt.figure(figsize = (5, 5), dpi = 125)

plt.plot(histarr[0]['acc'], marker='o')

plt.title('Score per epoch'); plt.ylabel('Validation Accuracy')
plt.xlabel('epoch')
plt.legend(labels = model_names[index]), plt.grid()
plt.savefig(ModelNamesArray + '_' + str(itera) + '-' + sche_name +'-Validation_Accuracy.jpg', bbox_inches = 'tight')

# Para guardar en un solo excel la historia de entrenamiento y los resultaddos del testing
excel_name = '%s-%s_%s-Loss_score_acc.xlsx' % (ModelNamesArray, sche_name, itera)
save_loss_score_acc(histarr, model_names[index], excel_name) # El ??ltimo valor es el nombre del fichero excel de comparativa de modelos


models_loaded = []
t_model = torch.load(f'./{model_names[index]}_best_model_{str(itera)}_itera.pt', weights_only = False)

'''
mob_miou = []
mob_acc = []
mob_miou = miou_score(t_model, test_set)

mob_miou_arr = np.array(mob_miou)
print(model_names[index])
#print(mob_miou)
m_miou = np.mean([x['val'] for x in mob_miou])
print('Test Set mIoU: ', m_miou)
miou_ = {'val': m_miou, 'img': ''}
mob_miou.insert(0,miou_)
excel_name = '%s_%s-img_iou_scores.xlsx' % (model_names[index], itera)
save_scores_imgs(mob_miou, model_names[index], excel_name,'IoU por fichero') 

mob_acc = pixel_acc(t_model, test_set)
mob_acc_arr = np.array(mob_acc)
m_acc = np.mean([x['val'] for x in mob_acc])
print('Test Set Acc: ', m_acc )
acc_ = {'val': m_acc, 'img': ''}
mob_acc.insert(0,acc_)
excel_name = '%s_%s-img_Acc_scores.xlsx' % (model_names[index], itera)
save_scores_imgs(mob_acc, model_names[index], excel_name,'Acc por fichero') 
'''
matrix = np.zeros((n_classes, n_classes))
t_model.eval()    
for i in range(1, len(test_set)):
    image, mask, fimg = test_set[i]
    pred_mask, score = predict_image_mask_miou(t_model, image, mask)
    target = np.array(mask)
    prediction = np.array(pred_mask)
    rows = target.shape[0]
    cols = target.shape[1]
    for i in range(rows):
        for j in range(cols):
            target_pixel = target[i][j]
            predict_pixel = prediction[i][j]      
            matrix[predict_pixel][target_pixel] += 1 

max_value, min_value = 0, 0
for i in range(n_classes):
    for j in range(n_classes):
        max_value = max(max_value, matrix[i][j])
        min_value = min(min_value, matrix[i][j])
matrix = pd.DataFrame(matrix)
#plt.figure(figsize = (12, 12), dpi = 125)
#sns.heatmap(matrix, annot = True, fmt=".0f", linewidths=.5)
save_confusion_matrix(matrix, model_names[index], f'{model_names[index]}_{str(itera)}_confusion_matrix.xlsx') # ultimo par??metro es el nombre del archivo con la matriz de confusi??n num??rica   
#plt.ylabel('Classes');
#plt.xlabel(f'{model_names[index]}_{str(itera)}')
#plt.savefig(f'{model_names[index]}_{str(itera)}-{sche_name}_confusion_matrix_table.jpg', bbox_inches = 'tight')
