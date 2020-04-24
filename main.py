from lib.engine import get_model_optim, train, eval, save_model
from lib.datasets import ADNI_dataset
import argparse
import torch


parser = argparse.ArgumentParser()
method = parser.add_argument("--method", type=str, default="")
epochs = parser.add_argument("--epochs", type=int, default=10)
batch_size = parser.add_argument("--batch-size", type=int, default=4)
data_split_ratio = parser.add_argument("--data-split-ratio", type=float, nargs="+")
data_path = parser.add_argument("--data-path", type=str, default="")
arg = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_epochs(train_ratio, val_ratio, test_ratio=0.0, crossval_fold=None):
    model, optim = get_model_optim(arg.method, device)

    train_dataset = ADNI_dataset(arg.method, arg.data_path, mode='Train', split_ratios=arg.data_split_ratio)
    val_dataset = ADNI_dataset(arg.method, arg.data_path, mode='Val', split_ratios=arg.data_split_ratio)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=arg.batch_size, \
        shuffle=True, num_workers=5, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=arg.batch_size, \
        shuffle=True, num_workers=5, pin_memory=True, drop_last=True)
    best_acc, best_loss = 0.0, 9.9
    for epoch in range(arg.epochs):
        train_loss, train_acc = train(model, train_loader, optim)
        val_loss, val_acc = eval(model, val_loader)
        print(epoch, train_loss, train_acc, val_loss, val_acc)
        if val_loss < best_loss:
            save_model(model, best_loss=True, checkpt_fname=arg.method)
            best_loss = val_loss

    if test_ratio > 0:
        test_dataset = ADNI_dataset(arg.method, arg.data_path, mode='Test', split_ratios=arg.data_split_ratio)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=arg.batch_size, \
        shuffle=True, num_workers=5, pin_memory=True, drop_last=True)
        checkpoint = torch.load("saved_model/"+arg.method+'_best_loss.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        test_loss, test_acc = eval(model, test_loader)
        print("Final testset performance: " + str(test_acc))
    

def cross_val(train_ratio, val_ratio, test_ratio=0.0, nfold=0):
    for fold_i in range(nfold):
        run(train_ratio, val_ratio, test_ratio, fold_i)

if arg.method == "liu":
    train_ratio, val_ratio, test_ratio = arg.data_split_ratio
    run_epochs(train_ratio, val_ratio, test_ratio)
elif arg.method == "jain":
    train_ratio, val_ratio = arg.data_split_ratio
    run_epochs(train_ratio, val_ratio)
elif arg.method == "soes":
    pass

