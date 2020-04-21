# import os and define graphic card
import os

os.environ["OMP_NUM_THREADS"] = "1"

# import common libraries
import random
import argparse
import pandas as pd
import numpy as np

# import pytorch related libraries
import torch
from tensorboardX import SummaryWriter
from transformers import *

# import apex for mix precision training
from apex import amp
amp.register_half_function(torch, "einsum")

# import dataset class
from dataset.dataset import *

# import utils
from utils.squad_metrics import *
from utils.ranger import *
from utils.lrs_scheduler import *
from utils.loss_function import *
from utils.metric import *
from utils.file import *

# import model
from model.model import *

# import config
from config import *


############################################################################## Define Argument
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--fold', type=int, default=0, required=False, help="specify the fold for training")
parser.add_argument('--model_type', type=str, default="se_resnext50", required=False, help="specify the model type")
parser.add_argument('--seed', type=int, default=2020, required=False, help="specify the seed")
parser.add_argument('--batch_size', type=int, default=16, required=False, help="specify the batch size")
parser.add_argument('--accumulation_steps', type=int, default=1, required=False, help="specify the accumulation_steps")


############################################################################## seed All
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    os.environ['PYHTONHASHseed'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True


############################################################################## Class for Plant
class Plant():
    def __init__(self, config):
        super(Plant).__init__()
        self.config = config
        self.setup_logger()
        self.setup_gpu()
        self.load_data()
        self.prepare_train()
        self.setup_model()

    def setup_logger(self):
        self.log = Logger()
        self.log.open((os.path.join(self.config.checkpoint_folder, "train_log.txt")), mode='a+')

    def setup_gpu(self):
        # confirm the device which can be either cpu or gpu
        self.config.use_gpu = torch.cuda.is_available()
        self.num_device = torch.cuda.device_count()
        if self.config.use_gpu:
            self.config.device = 'cuda'
            if self.num_device <= 1:
                self.config.data_parallel = False
            elif self.config.data_parallel:
                torch.multiprocessing.set_start_method('spawn', force=True)
        else:
            self.config.device = 'cpu'
            self.config.data_parallel = False

    def load_data(self):
        self.log.write('\nLoading data...')

        get_train_val_split(data_path=self.config.data_path,
                            save_path=self.config.save_path,
                            n_splits=self.config.n_splits,
                            seed=self.config.seed,
                            split=self.config.split)

        self.test_data_loader = get_test_loader(data_path=self.config.data_path,
                                                batch_size=self.config.val_batch_size,
                                                num_workers=self.config.num_workers,
                                                transforms=self.config.val_transforms)

        self.train_data_loader, self.val_data_loader = get_train_val_loader(data_path=self.config.data_path,
                                                                            seed=self.config.seed,
                                                                            fold=self.config.fold,
                                                                            batch_size=self.config.batch_size,
                                                                            val_batch_size=self.config.val_batch_size,
                                                                            num_workers=self.config.num_workers,
                                                                            transforms=self.config.transforms,
                                                                            val_transforms=self.config.val_transforms)

    def prepare_train(self):
        # preparation for training
        self.step = 0
        self.epoch = 0
        self.finished = False
        self.valid_epoch = 0
        self.train_loss, self.valid_loss, self.valid_metric_optimal = float('-inf'), float('-inf'), float('-inf')
        self.writer = SummaryWriter()
        ############################################################################### eval setting
        self.eval_step = int(len(self.train_data_loader) * self.config.saving_rate)
        self.log_step = int(len(self.train_data_loader) * self.config.progress_rate)
        self.eval_count = 0
        self.count = 0

    def pick_model(self):
        # for switching model
        self.model = PlantModel(model_name=self.config.model_type, num_classes=4).to(self.config.device)

    def differential_lr(self):

        param_optimizer = list(self.model.named_parameters())

        prefix = "backbone"

        def is_backbone(n):
            return prefix in n

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        self.optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and is_backbone(n)],
             'lr': self.config.min_lr,
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not is_backbone(n)],
             'lr': self.config.lr,
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and is_backbone(n)],
             'lr': self.config.min_lr,
             'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not is_backbone(n)],
             'lr': self.config.lr,
             'weight_decay': 0.0}
        ]

    def prepare_optimizer(self):

        # differential lr for each sub module first
        self.differential_lr()

        # optimizer
        if self.config.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.optimizer_grouped_parameters, eps=self.config.adam_epsilon)
        elif self.config.optimizer_name == "Ranger":
            self.optimizer = Ranger(self.optimizer_grouped_parameters)
        elif self.config.optimizer_name == "AdamW":
            self.optimizer = torch.optim.AdamW(self.optimizer_grouped_parameters, eps=self.config.adam_epsilon)
        elif self.config.optimizer_name == "FusedAdam":
            self.optimizer = FusedAdam(self.optimizer_grouped_parameters,
                                       bias_correction=False)
        else:
            raise NotImplementedError

        # lr scheduler
        if self.config.lr_scheduler_name == "WarmupCosineAnealing":
            num_train_optimization_steps = self.config.num_epoch * len(self.train_data_loader) \
                                           // self.config.accumulation_steps
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=self.config.warmup_steps,
                                                             num_training_steps=num_train_optimization_steps)
            self.lr_scheduler_each_iter = True
        elif self.config.lr_scheduler_name == "WarmCosineAnealingRestart-v2":
            T = len(self.train_data_loader) // self.config.accumulation_steps * 20  # cycle
            self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, T_0=T, T_mult=1, eta_max=self.config.lr * 25,
                                                      T_up=T // 20, gamma=0.2)
            self.lr_scheduler_each_iter = True
        elif self.config.lr_scheduler_name == "WarmCosineAnealingRestart":
            num_train_optimization_steps = self.config.num_epoch * len(self.train_data_loader) \
                                           // self.config.accumulation_steps
            self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=self.config.warmup_steps,
                                                             num_training_steps=num_train_optimization_steps)
            self.lr_scheduler_each_iter = True
        elif self.config.lr_scheduler_name == "WarmRestart":
            self.scheduler = WarmRestart(self.optimizer, T_max=5, T_mult=1, eta_min=1e-6)
            self.lr_scheduler_each_iter = False
        elif self.config.lr_scheduler_name == "WarmupLinear":
            num_train_optimization_steps = self.config.num_epoch * len(self.train_data_loader) \
                                           // self.config.accumulation_steps
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=self.config.warmup_steps,
                                                             num_training_steps=num_train_optimization_steps)
            self.lr_scheduler_each_iter = True
        elif self.config.lr_scheduler_name == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6,
                                                                        patience=2, min_lr=1e-7)
            self.lr_scheduler_each_iter = False
        else:
            raise NotImplementedError

        # lr scheduler step for checkpoints
        if self.lr_scheduler_each_iter:
            self.scheduler.step(self.step)
        else:
            self.scheduler.step(self.epoch)

    def prepare_apex(self):
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

    def load_check_point(self):
        self.log.write('Model loaded as {}.'.format(self.config.load_point))
        checkpoint_to_load = torch.load(self.config.load_point, map_location=self.config.device)
        self.step = checkpoint_to_load['step']
        self.epoch = checkpoint_to_load['epoch']
        self.valid_metric_optimal = checkpoint_to_load['valid_metric_optimal']

        model_state_dict = checkpoint_to_load['model']
        if self.config.load_from_load_from_data_parallel:
            # model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
            # "module.model"
            model_state_dict = {k[13:]: v for k, v in model_state_dict.items()}

        if self.config.data_parallel:
            state_dict = self.model.model.state_dict()
        else:
            state_dict = self.model.state_dict()

        keys = list(state_dict.keys())

        for key in keys:
            if any(s in key for s in self.config.skip_layers):
                continue
            try:
                state_dict[key] = model_state_dict[key]
            except:
                print("Missing key:", key)

        if self.config.data_parallel:
            self.model.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        if self.config.load_optimizer:
            self.optimizer.load_state_dict(checkpoint_to_load['optimizer'])

    def save_check_point(self):
        # save model, optimizer, and everything required to keep
        checkpoint_to_save = {
            'step': self.step,
            'epoch': self.epoch,
            'valid_metric_optimal': self.valid_metric_optimal,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}

        save_path = self.config.save_point.format(self.step, self.epoch)
        torch.save(checkpoint_to_save, save_path)
        self.log.write('Model saved as {}.'.format(save_path))

    def setup_model(self):
        self.pick_model()

        if self.config.data_parallel:
            self.prepare_optimizer()

            if self.config.apex:
                self.prepare_apex()

            if self.config.reuse_model:
                self.load_check_point()

            self.model = torch.nn.DataParallel(self.model)

        else:
            if self.config.reuse_model:
                self.load_check_point()

            self.prepare_optimizer()

            if self.config.apex:
                self.prepare_apex()

    def count_parameters(self):
        # get total size of trainable parameters
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def show_info(self):
        # show general information before training
        self.log.write('\n*General Setting*')
        self.log.write('\nseed: {}'.format(self.config.seed))
        self.log.write('\nmodel: {}'.format(self.config.model_type))
        self.log.write('\ntrainable parameters:{:,.0f}'.format(self.count_parameters()))
        self.log.write("\nmodel's state_dict:")
        self.log.write('\ndevice: {}'.format(self.config.device))
        self.log.write('\nuse gpu: {}'.format(self.config.use_gpu))
        self.log.write('\ndevice num: {}'.format(self.num_device))
        self.log.write('\noptimizer: {}'.format(self.optimizer))
        self.log.write('\nreuse model: {}'.format(self.config.reuse_model))
        if self.config.reuse_model:
            self.log.write('\nModel restored from {}.'.format(self.config.load_point))
        self.log.write('\n')

    def train_op(self):
        self.show_info()
        self.log.write('** start training here! **\n')
        self.log.write('   batch_size=%d,  accumulation_steps=%d\n' % (self.config.batch_size,
                                                                       self.config.accumulation_steps))
        self.log.write('   experiment  = %s\n' % str(__file__.split('/')[-2:]))

        while self.epoch <= self.config.num_epoch:

            self.train_metrics = 0

            # update lr and start from start_epoch
            if (self.epoch >= 1) and (not self.lr_scheduler_each_iter) \
                    and (self.config.lr_scheduler_name != "ReduceLROnPlateau"):
                self.scheduler.step()

            self.log.write("Epoch%s\n" % self.epoch)
            self.log.write('\n')

            sum_train_loss = np.zeros_like(self.train_loss)
            sum_train = np.zeros_like(self.train_loss)

            # init optimizer
            torch.cuda.empty_cache()
            self.model.zero_grad()

            for tr_batch_i, (image, _, label) in enumerate(self.train_data_loader):

                rate = 0
                for param_group in self.optimizer.param_groups:
                    rate += param_group['lr'] / len(self.optimizer.param_groups)

                # set model training mode
                self.model.train()

                # set input to cuda mode
                image = image.to(self.config.device)
                label = label.to(self.config.device)
                prediction = self.model(image)
                loss = LSR()(prediction, label)

                # use apex
                if self.config.apex:
                    with amp.scale_loss(loss / self.config.accumulation_steps, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # adversarial training
                if self.config.adversarial:
                    self.model.attack()
                    outputs_adv = self.model(image)
                    loss_adv = outputs_adv[0]

                    # use apex
                    if self.config.apex:
                        with amp.scale_loss(loss_adv / self.config.accumulation_steps, self.optimizer) as scaled_loss_adv:
                            scaled_loss_adv.backward()
                            self.model.restore()
                    else:
                        loss_adv.backward()
                    self.model.restore()

                if ((tr_batch_i + 1) % self.config.accumulation_steps == 0):
                    if self.config.apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), max_norm=
                        self.config.max_grad_norm, norm_type=2)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm,
                                                       norm_type=2)
                    self.optimizer.step()
                    self.model.zero_grad()
                    # adjust lr
                    if (self.lr_scheduler_each_iter):
                        self.scheduler.step()

                    self.writer.add_scalar('train_loss_' + str(self.config.fold), loss.item(),
                                           (self.epoch - 1) * len(
                                               self.train_data_loader) * self.config.batch_size + tr_batch_i *
                                           self.config.batch_size)
                    self.step += 1

                # translate to predictions
                prediction = prediction.detach().cpu().numpy()
                label = label.detach().cpu().numpy()

                metrics = cal_score_train(label, prediction)
                # running mean
                self.train_metrics = (self.train_metrics * tr_batch_i + metrics) / (tr_batch_i + 1)

                l = np.array([loss.item() * self.config.batch_size])
                n = np.array([self.config.batch_size])
                sum_train_loss = sum_train_loss + l
                sum_train = sum_train + n

                # log for training
                if (tr_batch_i + 1) % self.log_step == 0:
                    train_loss = sum_train_loss / (sum_train + 1e-12)
                    sum_train_loss[...] = 0
                    sum_train[...] = 0

                    self.log.write(
                        'lr: %f train loss: %f train_avg_acc: %f\n' % (rate, train_loss[0], self.train_metrics))

                if (tr_batch_i + 1) % self.eval_step == 0:
                    self.evaluate_op()

            if self.count >= self.config.early_stopping:
                break

            self.epoch += 1

    def evaluate_op(self):

        self.eval_count += 1
        valid_loss = np.zeros(1, np.float32)
        valid_num = np.zeros_like(valid_loss)

        self.eval_metrics = np.zeros(6)
        self.eval_prediction = None
        self.eval_prediction_softmax = None
        self.eval_label = None
        self.eval_onehot_label = None

        with torch.no_grad():

            # init cache
            torch.cuda.empty_cache()
            for val_batch_i, (image, onehot_label, label) in enumerate(self.val_data_loader):

                # set model to eval mode
                self.model.eval()

                # set input to cuda mode
                image = image.to(self.config.device)
                onehot_label = onehot_label.to(self.config.device)
                label = label.to(self.config.device)

                prediction = self.model(image)
                loss = nn.CrossEntropyLoss()(prediction, label)

                self.writer.add_scalar('val_loss_' + str(self.config.fold), loss.item(), (self.eval_count - 1) * len(
                    self.val_data_loader) * self.config.val_batch_size + val_batch_i * self.config.val_batch_size)

                # translate to predictions
                prediction_softmax = torch.softmax(prediction, dim=1).detach().cpu().numpy()
                prediction = prediction.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                onehot_label = onehot_label.detach().cpu().numpy()

                if self.eval_prediction is None:
                    self.eval_prediction = prediction
                else:
                    self.eval_prediction = np.concatenate([self.eval_prediction, prediction], axis=0)

                if self.eval_prediction_softmax is None:
                    self.eval_prediction_softmax = prediction_softmax
                else:
                    self.eval_prediction_softmax = np.concatenate([self.eval_prediction_softmax, prediction_softmax],
                                                                  axis=0)

                if self.eval_label is None:
                    self.eval_label = label
                else:
                    self.eval_label = np.concatenate([self.eval_label, label], axis=0)

                if self.eval_onehot_label is None:
                    self.eval_onehot_label = onehot_label
                else:
                    self.eval_onehot_label = np.concatenate([self.eval_onehot_label, onehot_label], axis=0)

                l = np.array([loss.item() * self.config.val_batch_size])
                n = np.array([self.config.val_batch_size])
                valid_loss = valid_loss + l
                valid_num = valid_num + n

            self.eval_metrics = cal_score(self.eval_onehot_label, self.eval_label, self.eval_prediction,
                                          self.eval_prediction_softmax)

            valid_loss = valid_loss / valid_num
            mean_eval_metric = self.eval_metrics[5]

            self.log.write(
                'val loss: %f ' % (valid_loss[0]) +
                'val_avg_auc: %f val_healthy_auc: %f val_multiple_diseases_auc: %f val_rust_auc: %f val_scab_auc: %f '
                % (self.eval_metrics[0], self.eval_metrics[1], self.eval_metrics[2], self.eval_metrics[3],
                   self.eval_metrics[4]) + 'val_avg_acc: %f\n' % (self.eval_metrics[5]))


        if self.config.lr_scheduler_name == "ReduceLROnPlateau":
            self.scheduler.step(mean_eval_metric)

        if mean_eval_metric >= self.valid_metric_optimal:

            self.log.write('Validation metric improved ({:.6f} --> {:.6f}).  Saving model ...'.format(
                self.valid_metric_optimal, mean_eval_metric))

            self.valid_metric_optimal = mean_eval_metric
            self.save_check_point()

            self.count = 0

            val_df = pd.DataFrame({'healthy': self.eval_prediction_softmax[:, 0],
                                   'multiple_diseases': self.eval_prediction_softmax[:, 1],
                                   'rust': self.eval_prediction_softmax[:, 2],
                                   'scab': self.eval_prediction_softmax[:, 3]})
            val_df.to_csv(
                os.path.join(self.config.checkpoint_folder, "val_prediction_{}_{}.csv".format(self.config.seed,
                                                                                              self.config.fold)),
                index=False)

        else:
            self.count += 1

    def infer_op(self):

        # save csv
        submission = pd.read_csv(os.path.join(self.config.data_path, "sample_submission.csv"))

        all_results = np.zeros((len(submission), 4))

        with torch.no_grad():

            # init cache
            torch.cuda.empty_cache()

            for test_batch_i, (image, _, _) in enumerate(self.test_data_loader):

                # set model to eval mode
                self.model.eval()

                # set input to cuda mode
                image = image.to(self.config.device)

                prediction = self.model(image)
                prediction_softmax = torch.softmax(prediction, dim=1).detach().cpu().numpy()
                # prediction_softmax = prediction.detach().cpu().numpy()

                all_results[test_batch_i * self.config.val_batch_size : (test_batch_i+1) * self.config.val_batch_size] \
                    = prediction_softmax


        submission[['healthy', 'multiple_diseases', 'rust', 'scab']] = all_results

        submission.to_csv(os.path.join(self.config.checkpoint_folder, "submission_{}.csv".format(self.config.fold)),
                          index=False)

        return

    def ensemble_op(self, models=None, seeds=None):
        # save csv
        if models is None:
            models = []
        if seeds is None:
            seeds = []
        submission = pd.read_csv(os.path.join(self.config.data_path, "sample_submission.csv"))
        prediction = np.zeros((len(submission), 4))

        for i, model in enumerate(models):

            seed = seeds[i]

            for fold in range(5):
                file_folder = os.path.join("/media/jionie/my_disk/Kaggle/Plant/model", model + "/seed_{}".format(seed)
                                           + "/fold_{}".format(fold))
                file_path = os.path.join(file_folder, "submission_{}.csv".format(fold))
                file = pd.read_csv(file_path)
                prediction += file[['healthy', 'multiple_diseases', 'rust', 'scab']].values / (len(models) * 5)

        submission[['healthy', 'multiple_diseases', 'rust', 'scab']] = prediction
        submission.to_csv("submission.csv", index=False)
        return


if __name__ == "__main__":
    args = parser.parse_args()

    # update fold
    config = Config(args.fold, model_type=args.model_type, seed=args.seed, batch_size=args.batch_size,
                    accumulation_steps=args.accumulation_steps)
    seed_everything(config.seed)
    qa = Plant(config)
    qa.train_op()
    # qa.evaluate_op()
    # qa.infer_op()
    # qa.ensemble_op(models=["se_resnext50"], seeds=[1997])