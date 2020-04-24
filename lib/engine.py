from lib.models import liunet, jainnet
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import shutil

def get_model_optim(method, device):
    if method == "liu":
        model = liunet.LiuNet()
        optim = getattr(torch.optim,"SGD")(model.parameters(), lr=0.01)
        return model.to(device), optim
    if method == "jain":
        model = jainnet.JainNet()
        optim = getattr(torch.optim,"RMSprop")(model.parameters(), lr=0.0001)
        return model.to(device), optim

def train(model, data_loader, optim):
    losses = []
    logit_all = []
    target_all = []
    for i, data in enumerate(data_loader):
        x, y = data[:-1], data[-1]
        logit, loss = model.compute_logit_loss(x,y)
        logit_all.append(logit.data.cpu())
        target_all.append(y.data.cpu())
        losses.append(loss.cpu().item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    logit_all = torch.cat(logit_all).numpy()
    target_all = torch.cat(target_all).numpy()
    acc = accuracy_score(target_all, np.argmax(logit_all,1))
    return np.mean(losses), acc

def eval(model, data_loader):
    losses = []
    logit_all = []
    target_all = []
    for i, data in enumerate(data_loader):
        x, y = data[:-1], data[-1]
        logit, loss = model.compute_logit_loss(x, y)
        logit_all.append(logit.clone().detach().cpu())
        target_all.append(y.clone().detach().cpu())
        _, preds = torch.max(logit.cpu(), 1)
        losses.append(loss.cpu().item())
    logit_all = torch.cat(logit_all).numpy()
    target_all = torch.cat(target_all).numpy()
    acc = accuracy_score(target_all, np.argmax(logit_all,1))
    return np.mean(losses), acc

def save_model(model, best_loss=False, best_acc=False, checkpt_fname=""):
    filename = "saved_model/" + checkpt_fname + ".pth.tar"
    torch.save({'state_dict': model.state_dict()}, filename)
    if best_loss:
        shutil.copyfile(filename, filename.replace('.pth.tar','')+'_best_loss.pth.tar')
    if best_acc:
        shutil.copyfile(filename, filename.replace('.pth.tar','')+'_best_acc.pth.tar')
