import torch
from scipy import stats
import numpy as np
import models
import data_loader
import scipy.io as scio


class HyperIQASolver(object):
    """Solver for training and testing hyperIQA"""

    def __init__(self, config, path_train, path_test, train_idx, test_idx):

        self.epochs = config.epochs
        self.model_hyper = models.IQANetSDG().cuda()
        self.model_hyper.train(True)
        self.l3_loss = torch.nn.L1Loss().cuda()
        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params,
                                      self.model_hyper.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)
        train_loader = data_loader.DataLoader02(config.dataset, path_train, train_idx, batch_size=config.batch_size,
                                                istrain=True)
        test_loader = data_loader.DataLoader02(config.datasetT, path_test, test_idx, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        srcc_per_epoch = np.zeros((1, self.epochs), dtype=np.float)
        plcc_per_epoch = np.zeros((1, self.epochs), dtype=np.float)
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []
            bcnt = 0
            xx=next(iter(self.train_data))

            for img, label, _, _ in self.train_data:
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda())
                self.solver.zero_grad()
                self.model_hyper.train(True)
                pred = self.model_hyper(img)
                predQ = pred['Q']
                predP = pred['P']
                pred_scores += predP.detach().cpu().numpy().tolist()
                gt_scores = gt_scores + label.cpu().tolist()
                loss01 = self.l3_loss(predP.squeeze(), label.float().detach())
                loss02 = self.l3_loss(predQ.squeeze(), torch.mean(label.float().detach(), dim=1))
                loss = loss01 + loss02
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()
                bcnt += 1
                if bcnt % 1000 == 0:
                    srcc_15 = []
                    for xx in range(15):
                        pred = np.array(pred_scores)[:, xx]
                        gt = np.array(gt_scores)[:, xx]
                        srcc_15.append(stats.spearmanr(pred, gt)[0])
                    print(
                        'Train:\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' % (
                        srcc_15[0], srcc_15[1], srcc_15[2], srcc_15[3], srcc_15[4], srcc_15[5], srcc_15[6], srcc_15[7],
                        srcc_15[8], srcc_15[9], srcc_15[10], srcc_15[11], srcc_15[12], srcc_15[13], srcc_15[14]))
                    test_srcc, test_plcc = self.test(self.test_data, bcnt)
                    print('%d\t%4.3f\t\t%4.4f\t\t%4.4f' %
                          (bcnt, sum(epoch_loss) / len(epoch_loss), test_srcc, test_plcc))
                    epoch_loss = []
                    pred_scores = []
                    gt_scores = []
                    torch.save(self.model_hyper, 'Amodel%d.pkl' % bcnt)
                if bcnt == 5000:
                    print('Apply Equal Learning Ratio...')
                    lr = self.lr
                    self.lrratio = 1
                    self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                                  {'params': self.model_hyper.res.parameters(), 'lr': lr}
                                  ]
                    self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        return 0.0, 0.0

    def test(self, data, t):
        """Testing"""
        self.model_hyper.train(False)
        self.model_hyper.eval()
        pred_scores = []
        gt_scores = []
        cnt = 0
        for img, label in data:
            # Data.
            img = torch.tensor(img.cuda())
            label = torch.tensor(label.cuda())
            pred = self.model_hyper(img)

            predQ = pred['Q']
            pred_scores += predQ.detach().cpu().numpy().tolist()
            gt_scores += label.cpu().numpy().tolist()
            cnt += 1

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)
        pred_scores = np.where(np.isnan(pred_scores), np.zeros_like(pred_scores), pred_scores)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        scio.savemat('Agt%d.mat' % t, {'gt': gt_scores})
        scio.savemat('Apred%d.mat' % t, {'pred': pred_scores})

        self.model_hyper.train(True)

        return test_srcc, test_plcc
