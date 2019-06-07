import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from lib.loader import invalid_collate

class EngineBase(object):
    def __init__(self, config, tb, logger, **kwargs):
        self.config = config
        self.logger = logger
        self.tb = tb

        self.model = None
        self.dataset = None
        self.pretrain_optim = None
        self.train_optim = None

    def pretrain(self):
        model = model.to(device=self.device)
        model.train()

        self.dataset.load_split("all")

        loader_params = {
            "batch_size": config.pretrain_batch_size,
            "num_workers": config.num_workers,
            "collate_fn": invalid_collate,
            "shuffle": True
        }
        loader = DataLoader(self.dataset, **loader_params)

        for num_iter, (x, y) in enumerate(loader):
            self.pretrain_optim.zero_grad()

            x = x.to(device=device).float()
            output, hidden_rep = model(x, reconstruct=True)

            if type(model) == torch.nn.DataParallel:
                loss = model.module.reconstruction_loss(output, x, hidden_rep)
            else:
                loss = model.reconstruction_loss(output, x, hidden_rep)

            loss.backward()
            self.pretrain_optim.step()

            yield x.detach(), y, loss.detach(), output.detach()

    def train(self):
        model = self.model.to(device=self.device)
        model.train()

        self.dataset.load_split("train")

        loader_params = {
            "batch_size": config.train_batch_size,
            "num_workers": config.num_workers,
            "collate_fn": invalid_collate,
            "shuffle": True
        }
        loader = DataLoader(self.dataset, **loader_params)

        for num_iter, (x, y) in enumerate(loader):
            if len(x) < torch.cuda.device_count() * 2: # skip for BatchNorm1d
                continue

            self.train_optim.zero_grad()

            x = x.to(device=device).float()
            y = y.to(device=device).long()

            pred = model(x)

            if type(model) == torch.nn.DataParallel:
                loss = model.module.loss(pred, y)
            else:
                loss = model.loss(pred, y)

            loss.backward()
            self.train_optim.step()

            yield x.detach(), y.detach(), loss.detach(), pred.detach()

    def validate(self):
        model = self.model.to(device=device)
        model.eval()

        self.dataset.load_split("valid")

        loader_params = {
            "batch_size": config.validate_batch_size,
            "num_workers": config.num_workers,
            "collate_fn": invalid_collate
        }
        loader = DataLoader(self.dataset, **loader_params)

        with torch.no_grad():
            for num_iter, (x, y) in enumerate(loader):
                x = x.to(device=device).float()
                y = y.to(device=device).long()

                pred = model(x)

                if type(model) == torch.nn.DataParallel:
                    loss = model.module.loss(pred, y)
                else:
                    loss = model.loss(pred, y)

                yield x, y, loss, pred

    def test(self):
        model = model.to(device=device)
        model.eval()

        self.dataset.load_split("test")

        loader_params = {
            "batch_size": config.test_batch_size,
            "num_workers": config.num_workers,
            "collate_fn": invalid_collate
        }
        loader = DataLoader(self.dataset, **loader_params)

        with torch.no_grad():
            for num_iter, (x, y) in enumerate(loader):
                x = x.to(device=device).float()
                y = y.to(device=device).long()

                pred = model(x)

                if type(model) == torch.nn.DataParallel:
                    loss = model.module.loss(pred, y)
                else:
                    loss = model.loss(pred, y)

                yield x, y, loss, pred

    def save_current_model(self, path):
        model = self.model.cpu()

        if type(model) == torch.nn.DataParallel:
            torch.save(model.module.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)

    def setup_pretrain_optimizer(self, optimizer=None):
        """ Setups the pre-training optimizer for the engine.
        """
        assert self.model is not None, "Attempting to create optimizer before setting up the model."

        optim_params = {
            "lr": self.config.pretrain_optim_lr,
            "weight_decay": self.config.pretrain_optim_wd
        }

        self.logger.info(f"Using pre-training optimizer with the following parameters: {optim_params}")

        self.pretrain_optim = \
            optim.Adam(self.model.parameters(), **optim_params)

    def setup_train_optimizer(self, optimizer=None):
        """ Setups the training optimizer for the engine.
        """
        assert self.model is not None, "Attempting to create optimizer before setting up the model."

        optim_params = {
            "lr": self.config.train_optim_lr,
            "weight_decay": self.config.train_optim_wd
        }

        self.logger.info(f"Using training optimizer with the following parameters: {optim_params}")

        self.train_optim = \
            optim.Adam(self.model.parameters(), **optim_params)

    def setup_device(self, device=None):
        """ Setups the device (CPU vs GPU) for the engine.
        """
        if device is not None:
            self.logger.info(f"Using {device} for training.")
            self.device = device
        else:
            cuda_available = torch.cuda.is_available()
            use_gpu = self.config.use_gpu
            gpu_count = torch.cuda.device_count()

            if cuda_available and use_gpu and gpu_count > 0:
                self.logger.info(f"Using {gpu_count} GPU for training.")
                self.device = torch.device("cuda")
            else:
                self.logger.info(f"Using CPU for training.")
                self.device = torch.device("cpu")
