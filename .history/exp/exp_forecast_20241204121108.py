from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import random
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.nn import DataParallel
warnings.filterwarnings('ignore')


def smape(y_true, y_pred):
    """
    Computes the symmetric Mean Absolute Percentage Error (sMAPE).
    
    Parameters:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Predicted values.
        
    Returns:
        torch.Tensor: sMAPE value.
    """
    numerator = torch.abs(y_true - y_pred)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2
    smape = torch.mean(numerator / (denominator + 1e-8)) * 100  # Add epsilon to avoid division by zero
    return smape

def smape_np(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(numerator / (denominator + 1e-8)) * 100  # Add epsilon to avoid division by zero
    return smape

class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)
        
    def _build_model(self):
        self.pred_lens = [self.args.output_token_len]
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.ddp:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        elif self.args.dp:
            self.device = self.args.gpu
            model = DataParallel(model, device_ids=self.args.device_ids).to(self.device)
        else:
            self.device = self.args.gpu
            model = model.to(self.device)
            
        if self.args.adaptation:
            state_dict = torch.load(self.args.pretrain_model_path)['state_dict']
            # state_dict = dict([(k.replace('model.', 'module.'),v) for k,v in state_dict.items()])
            state_dict_new = {}
            for k,v in state_dict.items():
                if k.startswith('model.patch_embedding.value_embedding'):
                    k = k.replace('model.patch_embedding.value_embedding', 'embedding')
                elif k.startswith('model.patch_embedding.position_embedding'):
                    k = k.replace('model.patch_embedding.position_embedding', 'position_embedding')
                else:
                    k = k.replace('model.', '')
                if self.args.dp:
                    k = 'module.' + k
                state_dict_new[k] = v
            model.load_state_dict(state_dict_new, strict=False)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = [[] for _ in range(len(self.pred_lens))]
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                B,O,C = batch_y.shape
                # mask = (torch.arange(C).expand(B, C).to(self.device) < lengths.unsqueeze(1).to(self.device)).unsqueeze(1)
                
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                batch_y = batch_y[:, :, :].to(self.device)
                # if is_test or self.args.nonautoregressive:
                #         outputs = outputs[:, -self.args.output_token_len:, :]
                #         batch_y = batch_y[:, -self.args.output_token_len:, :].to(self.device)
                # else:
                #     outputs = outputs[:, :, :]
                #     batch_y = batch_y[:, :, :].to(self.device)

                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                batch_y = batch_y[:, -self.args.output_token_len:, :].to(self.device)
                for length,output,tl in zip(self.pred_lens,outputs,total_loss):
                    loss = ((criterion(output[:, -length:, :], batch_y[:,:length,:])).mean()).detach().cpu()
                    # loss = smape(batch_y[:,:length,:], output[:, -length:, :]).item()
                    tl.append(loss)
                # loss = criterion(outputs, batch_y)

                # loss = loss.detach().cpu()
                # total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
        if self.args.ddp:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            # total_loss = np.average(total_loss, weights=total_count)
            total_loss = [np.average(tl, weights=total_count, axis=0) for tl in total_loss]
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        train_criterion = nn.MSELoss(reduction='none')
        criterion = self._select_criterion()
        
        # weights = torch.linspace(1, 0.5, steps=720).repeat(7)[None,:,None].to(self.device)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            # train_loader.dataset.__shuffle_data__()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # lengths = lengths.long().to(self.device)
                
                # lookback = random.randint(1,min(7,epoch+4))*self.args.input_token_len
                # batch_x = batch_x[:,-lookback:,:]
                # batch_y = batch_y[:,-lookback:,:]
                
                # B,S,M = batch_x.shape
                # indices = torch.randperm(M).to(batch_x.device)
                # batch_x = batch_x.index_select(-1, indices)
                # batch_y = batch_y.index_select(-1, indices)
                # gnum = 12
                # gsize = M//gnum
                # batch_x = batch_x[...,:gsize*gnum].reshape(B,-1,gnum,gsize).permute(0,2,1,3).reshape(B*gnum,-1,gsize)
                # batch_y = batch_y[...,:gsize*gnum].reshape(B,-1,gnum,gsize).permute(0,2,1,3).reshape(B*gnum,-1,gsize)
                B,O,C = batch_y.shape
                # mask = (torch.arange(C).expand(B, C).to(self.device) < lengths.unsqueeze(1)).unsqueeze(1)
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                if self.args.dp:
                    torch.cuda.synchronize()
                if self.args.nonautoregressive:
                    batch_y = batch_y[:, -self.args.output_token_len:, :]
                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                loss = 0
                
                patch_num = 7
                batch_y = batch_y.reshape(B,patch_num,-1,C)
                for output in outputs:
                    loss += (train_criterion(output, batch_y[:,:,:output.shape[1]//patch_num,:].reshape(B,-1,C))).mean()
                # loss = (train_criterion(outputs, batch_y) * weights).mean()
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                loss.backward()
                model_optim.step()
                
                # if (i + 1) % 5000 == 0:
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali(vali_data, vali_loader, train_criterion, is_test=self.args.valid_last)
            test_loss = self.vali(test_data, test_loader, train_criterion, is_test=True)
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {}, Steps: {} | Vali Loss: {} Test Loss: {}".format(
                    epoch + 1, train_steps, vali_loss, test_loss))
            early_stopping(sum(vali_loss), self.model, path)
            if early_stopping.early_stop:
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.ddp:
                train_loader.sampler.set_epoch(epoch + 1)
                
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.ddp:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.input_token_len, self.args.output_token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, best_model_path)))
        preds = [[] for _ in range(4)]
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        inference_steps = self.args.test_pred_len // self.args.output_token_len
        dis = self.args.test_pred_len - inference_steps * self.args.output_token_len
        if dis != 0:
            inference_steps += 1
        smape_avg = 0
        count = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # pred_y = []
                # for j in range(inference_steps):  
                #     if len(pred_y) != 0:
                #         batch_x = torch.cat([batch_x[:, self.args.input_token_len:, :], pred_y[-1]], dim=1)
                #     outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                #     pred_y.append(outputs[:, -self.args.output_token_len:, :])
                # pred_y = torch.cat(pred_y, dim=1)
                # pred_y = self.model(batch_x, batch_x_mark, None, batch_y_mark, inference_steps)
                # if dis != 0:
                #     pred_y = pred_y[:, :-dis, :]
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                batch_y = batch_y[:, -self.args.output_token_len:, :].detach().cpu()
                for length,output,pred in zip(self.pred_lens,outputs,preds):
                    pred.append(output[:, -length:, :].detach().cpu())
                    smape_cur = smape(batch_y[:,:pred[-1].shape[1],:],pred[-1])
                    smape_avg = (smape_avg*count + smape_cur*batch_y.shape[0])/(count+batch_y.shape[0])
                    count += batch_y.shape[0]
                # pred_y = pred_y[:, :self.args.test_pred_len, :]
                # batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                
                # outputs = pred_y.detach().cpu()
                
                # pred = outputs
                true = batch_y

                # preds.append(pred)
                trues.append(true)
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s, smape:{}".format(i + 1, speed, left_time, smape_avg))
                        iter_count = 0
                        time_now = time.time()
                if self.args.visualize and i % 2 == 0:
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    gt = np.array(true[0, :, -1])
                    pd = np.array(pred[0, :, -1])
                    visual(gt, pd, os.path.join(dir_path, f'{i}.pdf'))
        print(smape_avg)
        preds = [torch.cat(pred, dim=0).numpy() for pred in preds]
        trues = torch.cat(trues, dim=0).numpy()
        # print('preds shape:', preds.shape)
        print('trues shape:', trues.shape)
        
        if self.args.covariate:
            preds = preds[:, :, -1]
            trues = trues[:, :, -1]
        maes, mses = [], []
        for pred in preds:
            mae, mse, rmse, mape, mspe = metric(pred, trues[:,:pred.shape[1],:])
            # smape = smape_np(trues[:,:pred.shape[1],:],pred)
            maes.append(mae)
            mses.append(mse)
            # smapes.append(smape)
        print('mse:{}, mae:{}'.format(mses, maes))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mses, maes))
        f.write('\n')
        f.write('\n')
        f.close()
        return
