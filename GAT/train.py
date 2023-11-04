import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from model import AttentionModel
from baseline import RolloutBaseline
from dataset import generate_data, Generator
from config import Config, load_pkl, train_parser
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train(cfg):
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    model = AttentionModel(cfg.embed_dim, cfg.n_encode_layers, cfg.n_heads, cfg.tanh_clipping,
                           gru_dim = cfg.n_car_each_depot*cfg.n_depot*(cfg.n_customer+cfg.n_depot), cars=cfg.n_car_each_depot*cfg.n_depot)
    if num_gpus > 1:

        model = nn.DataParallel(model)
    print("Using", num_gpus, "GPUs!")
    model.to(device)
    baseline = RolloutBaseline(model, cfg.task, cfg.weight_dir, cfg.n_rollout_samples,
                               cfg.embed_dim, cfg.n_car_each_depot, cfg.n_depot, cfg.n_customer, cfg.capa,
                               cfg.warmup_beta, cfg.wp_epochs, device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    validation_dataset = Generator(device, n_samples=cfg.n_val_samples, n_car_each_depot=cfg.n_car_each_depot,
                                   n_depot=cfg.n_depot, n_customer=cfg.n_customer, seed=cfg.seed)
    val_L = baseline.validate(model, validation_dataset, 3 * cfg.batch)
    if cfg.islogger:
        val_path = '%s%s_%s_val.csv' % (cfg.log_dir, cfg.task, cfg.dump_date)  # cfg.log_dir = ./Csv/
        print(f'generate {val_path}')
        with open(val_path, 'w') as f:
            f.write('epoch,validation_cost\n')
            f.write('0,%1.4f\n' % (val_L))

    def rein_loss(model, inputs, bs, t, device):
        model.train()
        L, ll = model(inputs, decode_type='sampling')
        b = bs[t] if bs is not None else baseline.eval(inputs, L)
        return ((L - b.to(device)) * ll).mean(), L.mean()

    def get_results(train_loss_results, train_cost_results, val_cost, filename=None, plots=True):

        epochs_num = len(train_loss_results)

        df_train = pd.DataFrame(data={'epochs': list(range(epochs_num)),
                                      'loss': train_loss_results,
                                      'cost': train_cost_results,
                                      })
        df_test = pd.DataFrame(data={'epochs': list(range(epochs_num)),
                                     'val_сost': val_cost})
        data = {'epochs': list(range(epochs_num)),
                'loss': train_loss_results,
                'cost': train_cost_results,
                'val_cost': val_cost}

        df = pd.DataFrame(data)
        #df.to_csv('learning_scalewithoutgru_{}.csv'.format(cfg.n_customer), index=False)
        if plots:
            plt.figure(figsize=(15, 9))
            plt.gca().set_facecolor('whitesmoke')
            ax = sns.lineplot(x='epochs', y='loss', data=df_train, color='salmon', label='train loss')
            ax2 = ax.twinx()
            sns.lineplot(x='epochs', y='cost', data=df_train, color='cornflowerblue', label='train cost', ax=ax2)
            sns.lineplot(x='epochs', y='val_сost', data=df_test, palette='darkblue', label='val cost').set(
                ylabel='cost')
            ax.legend(loc=(0.75, 0.8), ncol=1, fontsize=20)
            ax2.legend(loc=(0.75, 0.87), ncol=1, fontsize=20)
            ax.grid(axis='x')
            ax2.grid(False)
            ax.grid(False)
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            # 设置坐标轴标签字号
            ax.set_xlabel('Epochs', fontsize=18)
            ax.set_ylabel('Train Loss', fontsize=18)
            ax2.set_ylabel('Val Cost', fontsize=18)
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
            ax2.tick_params(axis='y', labelsize=16)
            plt.savefig('learning_curve_plotwithgru_{}.jpg'.format(filename))
            plt.show()

    cnt = 0
    min_L = val_L
    val_cost_avg = []
    train_loss_results = []
    train_cost_results = []
    t1 = time()
    board_path = 'n%s_d%s_%s' % (cfg.n_customer, cfg.n_depot, datetime.now().strftime('%m_%d_%H_%M'))
    writer = SummaryWriter('runs/%s' % board_path)

    #dataset = Generator(device, cfg.n_samples, cfg.n_car_each_depot, cfg.n_depot, cfg.n_customer, cfg.seed)
    for epoch in range(cfg.epochs):
        avg_loss, avg_L, val_L = [0. for _ in range(3)]
        dataset = Generator(device, cfg.n_samples, cfg.n_car_each_depot, cfg.n_depot, cfg.n_customer, cfg.seed)
        bs = baseline.eval_all(dataset)
        bs = bs.view(-1, cfg.batch) if bs is not None else None  # bs: (cfg.batch_steps, cfg.batch) or None
        dataloader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True, drop_last=True)
        avg_loss_1 = 0
        avg_L_1 = 0
        for t, inputs in enumerate(tqdm(dataloader, disable=False)):
            # Assuming `inputs` is a batch of input data
            #if t == 0 and epoch == 0:
                #writer.add_graph(model, inputs)

            loss, L_mean = rein_loss(model, inputs, bs, t, device)
            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
            # print(f'loss: {loss.item()}')
            avg_loss += loss.item()
            # print('avg_loss',avg_loss)
            avg_L += L_mean.item()
            avg_loss_1 = avg_loss / (t + 1)
            avg_L_1 = avg_L / (t + 1)
            if t % (cfg.batch_verbose) == 0:
                t2 = time()
                print('Epoch %d (batch = %d): Loss: %1.3f L: %1.3f, %dmin%dsec' % (
                    epoch, t, avg_loss / (t + 1), avg_L / (t + 1), (t2 - t1) // 60, (t2 - t1) % 60))
                if cfg.islogger:
                    if t == 0 and epoch == 0:
                        log_path = '%s%s_%s.csv' % (cfg.log_dir, cfg.task, cfg.dump_date)  # cfg.log_dir = ./Csv/
                        print(f'generate {log_path}')
                        with open(log_path, 'w') as f:
                            f.write('time,epoch,batch,loss,cost\n')
                    with open(log_path, 'a') as f:
                        f.write('%dmin%dsec,%d,%d,%1.3f,%1.3f\n' % (
                            (t2 - t1) // 60, (t2 - t1) % 60, epoch, t, avg_loss / (t + 1), avg_L / (t + 1)))
                t1 = time()
        train_loss_results.append(avg_loss_1)
        train_cost_results.append(avg_L_1)
        baseline.epoch_callback(model, epoch, 3 * cfg.batch)
        val_L = baseline.validate(model, validation_dataset, 3 * cfg.batch)
        val_cost_avg.append(np.round(val_L.detach().cpu().numpy()))
        writer.add_scalar('Training Loss', avg_loss_1, epoch)
        writer.add_scalar('Training Cost', avg_L_1, epoch)
        writer.add_scalar('Validation Cost', val_L.item(), epoch)
        #for name, param in model.named_parameters():
            #writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        if cfg.islogger:
            with open(val_path, 'a') as f:
                f.write('%d,%1.4f\n' % (epoch, val_L))

        if (val_L < min_L):
            # model save
            weight_path = '%s%s_epoch%s_sample%s.pt' % (cfg.weight_dir, cfg.task, epoch, cfg.n_samples)
            if num_gpus > 1:
                torch.save(model.module.state_dict(), weight_path)

            else:
                torch.save(model.state_dict(), weight_path)
            print(f'update min val cost, {min_L}-->{val_L}\ngenerate {weight_path}')
            min_L = val_L
        else:
            cnt += 1
            print(f'cnt: {cnt}/10')
            if (cnt >= 30):
                print('early stop, average val cost cant decrease anymore')
                break

        if epoch == 0:
            if cfg.islogger:
                param_path = '%s%s_%s_param.csv' % (cfg.log_dir, cfg.task, cfg.dump_date)  # cfg.log_dir = ./Csv/
                print(f'generate {param_path}')
                with open(param_path, 'w') as f:
                    f.write(''.join('%s,%s\n' % item for item in vars(cfg).items()))
    filename_for_results = 'n_{}, d_{}, sample{}'.format(cfg.n_customer, cfg.n_depot,cfg.n_samples)
    writer.close()

    get_results(train_loss_results,
                train_cost_results,
                val_cost_avg,
                filename=filename_for_results,
                plots=True)


if __name__ == '__main__':
    cfg = load_pkl(train_parser().path)
    train(cfg)
