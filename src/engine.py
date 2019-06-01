import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from random import randint

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision.transforms as T

from normalized_dataset import NormalizedDataset
from models.vanilla_cae import VanillaCAE
from models.transform_cae import SpatialTransformConvAutoEnc
from models.classifier import Classify
from models.hosseini import Hosseini, HosseiniThreeLayer, HosseiniSimple, HosseiniDeep
from models.ae_cnn import AE_CNN
from models.two_d import TwoD
from models.deep_mri import DeepMRI
from models.deep_ae_mri import DeepAutoencMRI
from models.dense_net_3D import DenseNet3D
from models.pretrained import PretrainModel
from utils.transforms import RangeNormalization, NaNToNum, PadToSameDim, MeanStdNormalization

from utils.loader import invalid_collate

import os
import pickle

from pdb import set_trace

class Engine:
    OPTIMIZERS = ["adam", "sgd"]

    def __init__(self, config, tb_writer, logger, cross_val_flag, **kwargs):
        device = kwargs.get("device", None)
        model_path = kwargs.get("model_path", None)

        self.writer = tb_writer
        self.logger = logger
        self._config = config
        self._setup_device(device)
        self.load_model(model_path)
        self._setup_data(cross_val_flag)
 
        self.inception_mode = config["model"]["class"] == "inception_v3"

        self.train_optim = None
        self.train_scheduler = None
    """
    def pretrain(self, epoch):
        device = self._device
        config = self._config
        model = self._model

        print_iter = config["pretrain"]["print_iter"]
        model = model.to(device=device)
        model.train()

        optim_params = {
            "lr": config["pretrain"]["optim"]["learn_rate"],
            "weight_decay": config["pretrain"]["optim"]["weight_decay"],
            "momentum": config["pretrain"]["optim"]["momentum"]
        }
        optim_name = config["pretrain"]["optim"]["name"]

        if optim_name == "adam":
            del optim_params["momentum"]
            optimizer = optim.Adam(model.parameters(), **optim_params)
        elif optim_name == "sgd":
            optimizer = optim.SGD(model.parameters(), **optim_params)
        else:
            raise Exception("Unrecognized optimizer {}, valid values are {}"
                    .format(optim_name, self.OPTIMIZERS))

        losses = []

        for num_iter, (x, y) in enumerate(self.pretrain_loader):
            optimizer.zero_grad()

            x = x.to(device=device).float()

            output, hidden_rep = model(x, reconstruct=True)

            if type(model) == torch.nn.DataParallel:
                loss = model.module.reconstruction_loss(output, x, hidden_rep)
            else:
                loss = model.reconstruction_loss(output, x, hidden_rep)

            loss.backward()
            optimizer.step()

            output = output.detach()

            outputs = [
                output[:, idx, :, :, :] for idx in range(output.shape[1])
            ]

            if randint(1, 100) <= 10:
                # randomly pick one image to write to tensorboard
                img_idx = randint(0, len(output) - 1)
                # pick the middle slice
                mid_idx = output.shape[-1] // 2
                for idx, o in enumerate(outputs):
                    self.writer.add_image(
                        "Reconstruction/Channel {}/Coronal View".format(idx+1),
                        o[img_idx, mid_idx, :, :].unsqueeze(0),
                        global_step=epoch, dataformats="CHW")
                    self.writer.add_image(
                        "Reconstruction/Channel {}/Sagital View".format(idx+1),
                        o[img_idx, :, mid_idx, :].unsqueeze(0), global_step=epoch, dataformats="CHW")
                    self.writer.add_image(
                        "Reconstruction/Channel {}/Axial View".format(idx+1),
                        o[img_idx, :, :, mid_idx].unsqueeze(0), global_step=epoch, dataformats="CHW")

            losses.append(loss.item())
            if num_iter % print_iter == 0:
                print("\tIteration {} ({}): {}"
                            .format(num_iter, y.detach(), loss.item()))

        return {
            "loss_history": losses,
            "average_loss": sum(losses) / len(losses)
        }
    """

    def train(self, epoch=0, max_epochs=33):
        '''
        Execute the training loop.

        Args:
            **kwargs:
                print_iter (int): How often (number of iterations) to print output.
        '''
        device = self._device
        config = self._config
        model = self._model

        print_iter = config["train"]["print_iter"]

        model = model.to(device=device)
        model.train()

        if config["train"]["freeze_cnn"]:
            self.logger.log("Training with CNN weights frozen.")
            if type(model) == torch.nn.DataParallel:
                model.module.freeze()
            else:
                model.freeze()

        if self.train_optim is None:
            self._setup_train_optimizer()

        if self.train_scheduler is not None:
            #self.train_scheduler.step()
            for param_group in self.train_optim.param_groups:
                lr_orig = self._config["train"]["optim"]["learn_rate"]
                param_group['lr'] = lr_orig * (1 - epoch / max_epochs)**0.5
                print(param_group['lr'])

        losses = []
        tally = {
            # [ num correct, total ]
            "AD": [0, 0],
            "CN": [0, 0],
            "MCI": [0, 0],
            "Total": [0, 0]
        }
        for num_iter, (x, y) in enumerate(self.train_loader):
            if len(x) < self._gpu_count * 2: # skip for BatchNorm1d
                continue
       
            self.train_optim.zero_grad()

            if randint(1, 100) <= 5:
                temp_x = x.detach()
                # randomly pick one image to write to tensorboard
                img_idx = randint(0, len(temp_x) - 1)
                if len(temp_x.shape) == 5:
                    # enumerate all of the channels
                    inputs = [
                        temp_x[:, idx, :, :, :] for idx in range(x.shape[1])
                    ]
                    # pick the middle slice
                    mid_idx = temp_x.shape[-1] // 2
                    for idx, inp in enumerate(inputs):
                        self.writer.add_image(
                            "Training Input/Channel {}/Coronal View"
                                .format(idx+1),
                            inp[img_idx, mid_idx, :, :].unsqueeze(0),
                            global_step=epoch, dataformats="CHW")
                        self.writer.add_image(
                            "Training Input/Channel {}/Axial View"
                                .format(idx+1),
                            inp[img_idx, :, :, mid_idx].unsqueeze(0),
                            global_step=epoch, dataformats="CHW")
                        self.writer.add_image(
                            "Training Input/Channel {}/Sagital View"
                                .format(idx+1),
                            inp[img_idx, :, mid_idx, :].unsqueeze(0),
                            global_step=epoch, dataformats="CHW")
                elif len(x.shape) == 4:
                    self.writer.add_image(
                        "Training Input", temp_x[img_idx, :, :, :],
                        global_step=epoch, dataformats="CHW"
                    )

            x = x.to(device=device).float()
            y = y.to(device=device).long()

            pred = model(x)
            if self.inception_mode == True:
                pred = pred[0]
            if type(model) == torch.nn.DataParallel:
                loss = model.module.loss(pred, y)
            else:
                loss = model.loss(pred, y)
            if y.dim() == 1: # classification
                with torch.no_grad():
                    pred = pred.argmax(dim=1)

                    label_encoder = self.train_loader.dataset.label_encoder
                    correct = (y.cpu().numpy() == pred.cpu().numpy())
                    labels = label_encoder.inverse_transform(y.cpu().numpy())
                    tally = self._add_to_tally(labels, correct, tally)
            loss.backward()
            self.train_optim.step()
            losses.append(loss.item())
        return {
            "loss_history": losses,
            "average_loss": sum(losses) / len(losses)
        }, tally

    def validate(self):
        device = self._device
        config = self._config
        model = self._model

        model = model.to(device=device)
        model.eval()

        losses = []
        num_correct = 0
        num_total = 0
        tally = {
            # [ num correct, total ]
            "AD": [0, 0],
            "CN": [0, 0],
            "MCI": [0, 0],
            "Total": [0, 0]
        }

        with torch.no_grad():
            for num_iter, (x, y) in enumerate(self.valid_loader):

                x = x.to(device=device).float()
                y = y.to(device=device).long()

                pred = model(x)

                if type(model) == torch.nn.DataParallel:
                    loss = model.module.loss(pred, y)
                else:
                    loss = model.loss(pred, y)

                if y.dim() == 1: # classification
                    pred = pred.argmax(dim=1)
                    num_correct += (y == pred).sum().item()
                    num_total += len(y)

                    label_encoder = self.train_loader.dataset.label_encoder
                    correct = (y.cpu().numpy() == pred.cpu().numpy())
                    labels = label_encoder.inverse_transform(y.cpu().numpy())
                    tally = self._add_to_tally(labels, correct, tally)

                losses.append(loss.item())

        return {
            "loss_history": losses,
            "average_loss": sum(losses) / len(losses),
            "num_correct": num_correct,
            "num_total": num_total
        }, tally

    def test(self):
        device = self._device
        config = self._config
        model = self._model

        model = model.to(device=device)
        model.eval()

        losses = []
        num_correct = 0
        num_total = 0
        tally = {
            # [ num correct, total ]
            "AD": [0, 0],
            "CN": [0, 0],
            "MCI": [0, 0],
            "Total": [0, 0]
        }

        with torch.no_grad():
            for num_iter, (x, y) in enumerate(self.test_loader):
                x = x.to(device=device).float()
                y = y.to(device=device).long()

                pred = model(x)

                if type(model) == torch.nn.DataParallel:
                    loss = model.module.loss(pred, y)
                else:
                    loss = model.loss(pred, y)

                if y.dim() == 1: # classification
                    pred = pred.argmax(dim=1)
                    num_correct += (y == pred).sum().item()
                    num_total += len(y)

                    label_encoder = self.train_loader.dataset.label_encoder
                    correct = (y.cpu().numpy() == pred.cpu().numpy())
                    labels = label_encoder.inverse_transform(y.cpu().numpy())
                    tally = self._add_to_tally(labels, correct, tally)

                losses.append(loss.item())

        pct_correct = round((tally["Total"][0] * 100.0) / tally["Total"][1], 2)
        print("\tTest correct: AD {}/{}, CN {}/{}, MCI {}/{}, total {}/{}({}%)"
                .format(tally["AD"][0], tally["AD"][1], tally["CN"][0],
                        tally["CN"][1], tally["MCI"][0], tally["MCI"][1],
                        tally["Total"][0], tally["Total"][1], pct_correct))

        return {
            "loss_history": losses,
            "average_loss": sum(losses) / len(losses),
            "num_correct": num_correct,
            "num_total": num_total
        }, tally

    def gen_feature_outputs(self):
        device = self._device
        config = self._config
        model = self._model

        model = model.to(device=device)
        
        with torch.no_grad():
            
            def gen_feature_outputs_helper(data_loader):
                i = 0
                for num_iter, (x, y) in enumerate(data_loader):
                    i += 1
                    print(i)
                    print(y)
                    x = x.to(device=device).float()
                    pred = model(x)
                    if (self.inception_mode == True):
                        pred = pred[0]
                    for i in range(pred.shape[0]):
                        filepath = '/mnt/nfs/work1/mfiterau/ADNI_data/feature_output/resnet/'
                        if not os.path.exists(filepath + y[0][i]):
                            os.mkdir(filepath + y[0][i])
                        if not os.path.exists(filepath + y[0][i] + '/' + y[1][i]):
                            os.mkdir(filepath + y[0][i] + '/' + y[1][i])
                        filepath += y[0][i] + '/' + y[1][i] + '/features.pckl' 
                        pred = pred.cpu()
                        torch.save(pred[i], filepath)
                print(i)
            print("On training data right now...")
            gen_feature_outputs_helper(self.train_loader)
            print("On to validation data now...")
            gen_feature_outputs_helper(self.valid_loader)

    def save_model(self, path, **kwargs):
        model = self._model.cpu()

        if type(model) == torch.nn.DataParallel:
            torch.save(model.module.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)

    def _setup_device(self, device):
        '''
        Set the device (CPU vs GPU) used for the experiment. Defaults to GPU if available.

        Args:
            device (torch.device, optional): Defaults to None. If supplied, this will be used instead.
        '''
        if device is not None:
            self._device = device
            return

        cuda_available = torch.cuda.is_available()
        self._use_gpu = self._config["use_gpu"]
        self._gpu_count = torch.cuda.device_count()

        if cuda_available and self._use_gpu and self._gpu_count > 0:
            print("{} GPUs detected, running in GPU mode."
                    .format(self._gpu_count))
            self._device = torch.device("cuda")
        else:
            print("Running in CPU mode.")
            self._device = torch.device("cpu")

    def load_model(self, model_path=None):
        config = self._config
        model_class = config["model"]['class']
        n_channels = len(config["image_col"])

        if model_class == "deep_ae_mri":
            print("Using deep AE MRI model")
            self._model = DeepAutoencMRI(num_channels=n_channels,
                                num_blocks=config["model"]["num_blocks"],
                                sparsity=config["pretrain"]["sparsity"],
                                cnn_dropout=config["train"]["cnn_dropout"],
                                class_dropout=config["train"]["class_dropout"])
        elif model_class == "3D_dense_net":
            print("Using Wang et al. 3D DenseNet model")
            self._model = DenseNet3D(num_channels=n_channels,
                                num_blocks=config["model"]["num_blocks"],
                                sparsity=config["pretrain"]["sparsity"],
                                cnn_dropout=config["train"]["cnn_dropout"],
                                class_dropout=config["train"]["class_dropout"])
        elif model_class in ["resnet18", "densenet121", "vgg16", "alexnet", "inception_v3"]:
            print("Using {} model.".format(model_class))
            self._model = PretrainModel(model_name=model_class,
                            freeze_weight=self._config["train"]["freeze_cnn"],
                            pretrained=self._config["model"]["pretrained"],
                            weights=model_path,
                            save_features=self._config["features_to_pickles"])
        else:
            raise Exception("Unrecognized model: {}".format(model_class))

        if model_path is not None:
            self._model.load_state_dict(torch.load(model_path))

        if self._use_gpu and self._gpu_count > 1:
            self._model = nn.DataParallel(self._model)

    def _setup_data(self, cross_val_fold_num):
        config = self._config
        num_workers = min(mp.cpu_count() // 2, config["data"]["max_workers"])
        num_workers = max(num_workers, 1)

        transforms = self._get_transforms()
        pretrain_dataset_params = {
            "mode": "all",
            "task": "pretrain",
            "valid_split": 0.0,
            "test_split": 0.0,
            "limit": config["data"]["limit"],
            "config": config,
            "transforms": transforms
        }
        train_dataset_params = {
            "mode": "train",
            "task": "classify",
            "valid_split": config["data"]["valid_split"],
            "test_split": config["data"]["test_split"],
            "limit": config["data"]["limit"],
            "config": config,
            "transforms": transforms,
            "cross_val_fold": cross_val_fold_num,
            "features_to_pckl": config["features_to_pickles"]
        }
        valid_dataset_params = {
            "mode": "valid",
            "task": "classify",
            "valid_split": config["data"]["valid_split"],
            "test_split": config["data"]["test_split"],
            "limit": config["data"]["limit"],
            "config": config,
            "transforms": transforms,
            "cross_val_fold": cross_val_fold_num,
            "features_to_pckl": config["features_to_pickles"]
        }
        test_dataset_params = {
            "mode": "test",
            "task": "classify",
            "valid_split": config["data"]["valid_split"],
            "test_split": config["data"]["test_split"],
            "limit": config["data"]["limit"],
            "config": config,
            "transforms": transforms
        }

        pretrain_loader_params = {
            "batch_size": config["pretrain"]["batch_size"],
            "num_workers": num_workers,
            "collate_fn": invalid_collate,
            "drop_last": True,
            "shuffle": True
        }
        train_loader_params = {
            "batch_size": config["train"]["batch_size"],
            "num_workers": num_workers,
            "collate_fn": invalid_collate,
            "shuffle": True
        }
        valid_loader_params = {
            "batch_size": config["valid"]["batch_size"],
            "num_workers": num_workers,
            "collate_fn": invalid_collate,
            "shuffle": True
        }
        test_loader_params = {
            "batch_size": config["test"]["batch_size"],
            "num_workers": num_workers,
            "collate_fn": invalid_collate,
            "shuffle": True
        }

        if config["data"]["set_name"] == "normalized":
            num_dim = config["data"]["num_dim"]
            slice_view = config["data"]["slice_view"]
            slice_num = config["data"]["slice_num"]
            self.pretrain_dataset = NormalizedDataset(
                                        **pretrain_dataset_params)
            self.train_dataset = NormalizedDataset(
                                        **train_dataset_params)
            # Ensure the label and its encoded counter part match.
            label_encoder = self.train_dataset.label_encoder
            self.valid_dataset = NormalizedDataset(
                                        label_encoder=label_encoder,
                                        **valid_dataset_params)
            self.test_dataset = NormalizedDataset(
                                        label_encoder=label_encoder,
                                        **test_dataset_params)
        #self.pretrain_loader = DataLoader(self.pretrain_dataset,
        #                                  **pretrain_loader_params)
        ## Note by Yi: uneven batch may occur at the end when data_size % batch_size != 0
        ## See here: https://forums.fast.ai/t/understanding-code-error-expected-more-than-1-value-per-channel-when-training/9257/10
        self.train_loader = DataLoader(self.train_dataset, drop_last=True,
                                       **train_loader_params)
        self.valid_loader = DataLoader(self.valid_dataset, drop_last=True,
                                       **valid_loader_params)
       
        if cross_val_fold_num != -1:
            self.test_loader = DataLoader(self.test_dataset,
                                          **test_loader_params)

        print("{} training data, {} validation data, {} test data"
                .format(len(self.train_dataset),
                        len(self.valid_dataset),
                        len(self.test_dataset)))

    def _setup_train_optimizer(self):
        optim_params = {
            "lr": self._config["train"]["optim"]["learn_rate"],
            "weight_decay": self._config["train"]["optim"]["weight_decay"],
            "momentum": self._config["train"]["optim"]["momentum"]
        }
        optim_name = self._config["train"]["optim"]["name"]

        use_scheduler = self._config["train"]["optim"]["use_scheduler"]
        step_size = self._config["train"]["optim"]["step_size"]
        decay_factor = self._config["train"]["optim"]["decay_factor"]

        if optim_name == "adam":
            del optim_params["momentum"]
            optimizer = optim.Adam(self._model.parameters(), **optim_params)
        elif optim_name == "sgd":
            optimizer = optim.SGD(self._model.parameters(), **optim_params)
        else:
            raise Exception("Unrecognized optimizer {}, valid values are {}"
                    .format(optim_name, self.OPTIMIZERS))

        if use_scheduler is True:
            self.train_scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=step_size, gamma=decay_factor)

        self.train_optim = optimizer

    def _add_to_tally(self, labels, gt, tally):
        ad = labels == "AD"
        cn = labels == "CN"
        mci = labels == "MCI"
        tally["AD"][0] += (ad * gt).sum()
        tally["AD"][1] += ad.sum()
        tally["CN"][0] += (cn * gt).sum()
        tally["CN"][1] += cn.sum()
        tally["MCI"][0] += (mci * gt).sum()
        tally["MCI"][1] += mci.sum()
        tally["Total"][0] += gt.sum()
        tally["Total"][1] += len(labels)

        return tally

    def _get_transforms(self):
        '''Compile image transform functions.
        '''
        config = self._config

        transforms = []
        
        if config["data"]["num_dim"] == 2 and \
            not config["data"]["set_name"] == "presliced":
            transforms.append(T.ToPILImage("RGB"))

        # transforms below operate on PIL Image
        if "resize" in config["data"]["transforms"]:
             # 2D ONLY!
            if config["data"]["num_dim"] == 3:
                raise Exception("Attempting to call resize on 3D image.")
            size = config["data"]["transforms"]["resize"]
            transforms.append(T.Resize((size, size)))

        transforms.append(T.ToTensor())

        # transforms below operate on Pytorch Tensors
        if "pad_to_same" in config["data"]["transforms"] and \
            config["data"]["transforms"]["pad_to_same"]:
            transforms.append(PadToSameDim(config["data"]["num_dim"]))

        if "nan_to_num" in config["data"]["transforms"]:
            target = config["data"]["transforms"]["nan_to_num"]
            transforms.append(NaNToNum(target))

        if "range_norm" in config["data"]["transforms"] and \
            config["data"]["transforms"]["range_norm"]:
            transforms.append(RangeNormalization())
        elif "mean_std_norm" in config["data"]["transforms"] and \
            config["data"]["transforms"]["mean_std_norm"]:
            transforms.append(MeanStdNormalization())

        return transforms
