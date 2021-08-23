import torch
from scipy import stats
import numpy as np
import models
import data_loader
import scipy.io as scio


class HyperIQASolver(object):
    """Solver for training and testing hyperIQA"""

    def __init__(self, config, idx_config):
        self.rounds = config.rounds
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
        self.train_loader_cuz = data_loader.DataLoader02('cuz', idx_config['cuz'], batch_size=config.batch_size,
                                                         istrain=True)
        self.train_loader_cuzG = data_loader.DataLoader02('cuzG', idx_config['cuzG'], batch_size=8,
                                                          istrain=True)
        self.train_loader_cuzP = data_loader.DataLoader02('cuzP', idx_config['cuzP'], batch_size=8,
                                                          istrain=True)
        self.train_loader_cuzE = data_loader.DataLoader02('cuzE', idx_config['cuzEtr'], batch_size=config.batch_size,
                                                          istrain=True)
        self.train_loader_cuzEG = data_loader.DataLoader02('cuzEG', idx_config['cuzEG'], batch_size=8,
                                                           istrain=True)
        self.train_loader_cuzEP = data_loader.DataLoader02('cuzEP', idx_config['cuzEP'], batch_size=8,
                                                           istrain=True)
        testFR_loader = data_loader.DataLoader02('cuzE', idx_config['cuzEte'], batch_size=config.batch_size,
                                                          istrain=True)
        test_loader = data_loader.DataLoader02(config.datasetT, idx_config['test'], batch_size=1, istrain=False)
        self.train_data_cuz = iter(self.train_loader_cuz.get_data())
        self.train_data_cuzG = iter(self.train_loader_cuzG.get_data())
        self.train_data_cuzP = iter(self.train_loader_cuzP.get_data())
        self.train_data_cuzE = iter(self.train_loader_cuzE.get_data())
        self.train_data_cuzEG = iter(self.train_loader_cuzEG.get_data())
        self.train_data_cuzEP = iter(self.train_loader_cuzEP.get_data())
        self.test_data = (test_loader.get_data())
        self.testFR_data = (test_loader.get_data())
        self.MTH = 0.05

    def trainOnce(self, img, label):
        img = torch.tensor(img.cuda())
        label = torch.tensor(label.cuda())
        self.solver.zero_grad()
        self.model_hyper.train(True)
        pred = self.model_hyper(img)
        predQ = pred['Q']
        predP = pred['P']
        #predP2 = pred['P2']
        loss01 = self.l3_loss(predP.squeeze(), label.float().detach())
        loss02_tmp = self.l3_loss(predQ.squeeze(), torch.mean(label.float().detach(), dim=1))
        loss02 = torch.where(loss02_tmp < self.MTH, torch.zeros_like(loss02_tmp), loss02_tmp)

        #loss03 = self.l3_loss(predP.detach().squeeze(), predP2.squeeze())

        loss = loss01 + loss02
        loss.backward()
        self.solver.step()
        return predP, loss

    def trainOnceDual(self, imgG, labelG, imgP, labelP, TH):
        imgG = torch.tensor(imgG.cuda())
        labelG = torch.tensor(labelG.cuda())
        imgP = torch.tensor(imgP.cuda())
        labelP = torch.tensor(labelP.cuda())
        self.solver.zero_grad()
        self.model_hyper.train(True)
        pred_G = self.model_hyper(imgG)
        pred_P = self.model_hyper(imgP)

        predQ_G = pred_G['Q']
        predP_G = pred_G['P']
        #predP2_G = pred_G['P2']

        predQ_P = pred_P['Q']
        predP_P = pred_P['P']
        #predP2_P = pred_P['P2']

        loss01_G = self.l3_loss(predP_G.squeeze(), labelG.float().detach())
        loss01_P = self.l3_loss(predP_P.squeeze(), labelP.float().detach())

        loss02_G_tmp = self.l3_loss(predQ_G.squeeze(), torch.mean(labelG.float().detach(), dim=1))
        loss02_P_tmp = self.l3_loss(predQ_P.squeeze(), torch.mean(labelP.float().detach(), dim=1))

        loss02_G = torch.where(loss02_G_tmp < self.MTH, torch.zeros_like(loss02_G_tmp), loss02_G_tmp)
        loss02_P = torch.where(loss02_P_tmp < self.MTH, torch.zeros_like(loss02_P_tmp), loss02_P_tmp)

        #loss03_G = self.l3_loss(predP_G.detach().squeeze(),predP2_G.squeeze())
        #loss03_P = self.l3_loss(predP_P.detach().squeeze(), predP2_P.squeeze())

        loss04_tmp = torch.mean(predQ_P) + TH - torch.mean(predQ_G)
        loss04 = 0.0 if loss04_tmp < 0.0 else loss04_tmp

        #loss = loss01_G + loss01_P + 0.2*(loss02_G + loss02_P) + loss03_G + loss03_P + 2 * loss04
        loss = loss01_G + loss01_P + loss02_G + loss02_P + 0.5 * loss04
        loss.backward()
        self.solver.step()
        return predP_G, predP_P, loss

    def analysis(self, pred_scores, gt_scores):
        srcc_15 = []
        for xx in range(16):
            pred = np.array(pred_scores)[:, xx]
            gt = np.array(gt_scores)[:, xx]
            srcc_15.append(stats.spearmanr(pred, gt)[0])
        print(
            'Train:\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' % (
                srcc_15[0], srcc_15[1], srcc_15[2], srcc_15[3], srcc_15[4], srcc_15[5], srcc_15[6], srcc_15[7],
                srcc_15[8], srcc_15[9], srcc_15[10], srcc_15[11], srcc_15[12], srcc_15[13], srcc_15[14], srcc_15[15]))

    def analysisDual(self, pred_scores_G, pred_scores_P, gt_scores_G, gt_scores_P):
        srcc_15 = []
        for xx in range(16):
            pred_G = np.array(pred_scores_G)[:, xx]
            gt_G = np.array(gt_scores_G)[:, xx]
            srcc_15.append(stats.spearmanr(pred_G, gt_G)[0])
        print(
            'Train:\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' % (
                srcc_15[0], srcc_15[1], srcc_15[2], srcc_15[3], srcc_15[4], srcc_15[5], srcc_15[6], srcc_15[7],
                srcc_15[8], srcc_15[9], srcc_15[10], srcc_15[11], srcc_15[12], srcc_15[13], srcc_15[14], srcc_15[15]))
        srcc_15 = []
        for xx in range(16):
            pred_P = np.array(pred_scores_P)[:, xx]
            gt_P = np.array(gt_scores_P)[:, xx]
            srcc_15.append(stats.spearmanr(pred_P, gt_P)[0])
        print(
            'Train:\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' % (
                srcc_15[0], srcc_15[1], srcc_15[2], srcc_15[3], srcc_15[4], srcc_15[5], srcc_15[6], srcc_15[7],
                srcc_15[8], srcc_15[9], srcc_15[10], srcc_15[11], srcc_15[12], srcc_15[13], srcc_15[14], srcc_15[15]))

    def train(self):
        """Training"""
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        epoch_loss = []
        pred_scores = []
        gt_scores = []
        epoch_loss_Dual = []
        pred_scores_G = []
        pred_scores_P = []
        gt_scores_G = []
        gt_scores_P = []
        epoch_lossE = []
        pred_scoresE = []
        gt_scoresE = []
        epoch_loss_DualE = []
        pred_scores_GE = []
        pred_scores_PE = []
        gt_scores_GE = []
        gt_scores_PE = []

        for bcnt in range(self.rounds):
            # cuz
            for t in range(self.epochs):
                try:
                    img, label = self.train_data_cuz.next()
                except StopIteration:
                    self.train_data_cuz = iter(self.train_loader_cuz.get_data())
                    img, label = self.train_data_cuz.next()
                predP, loss = self.trainOnce(img, label)
                pred_scores += predP.detach().cpu().numpy().tolist()
                gt_scores = gt_scores + label.cpu().tolist()
                epoch_loss.append(loss.item())

            if (bcnt != 0) and (bcnt % 100 == 0):
                self.analysis(pred_scores, gt_scores)
                epoch_loss = []
                pred_scores = []
                gt_scores = []

            # cuzP & cuzG
            '''
                        for t in range(self.epochs//5):
                try:
                    imgG, labelG = self.train_data_cuzG.next()
                except StopIteration:
                    self.train_data_cuzG = iter(self.train_loader_cuzG.get_data())
                    imgG, labelG = self.train_data_cuzG.next()
                try:
                    imgP, labelP = self.train_data_cuzP.next()
                except StopIteration:
                    self.train_data_cuzP = iter(self.train_loader_cuzP.get_data())
                    imgP, labelP = self.train_data_cuzP.next()

                predP_G, predP_P, loss = self.trainOnceDual(imgG, labelG, imgP, labelP, 0.10)

                pred_scores_G += predP_G.detach().cpu().numpy().tolist()
                pred_scores_P += predP_P.detach().cpu().numpy().tolist()

                gt_scores_G = gt_scores_G + labelG.cpu().tolist()
                gt_scores_P = gt_scores_P + labelP.cpu().tolist()
                epoch_loss_Dual.append(loss.item())

            if (bcnt != 0) and (bcnt % 100 == 0):
                self.analysisDual(pred_scores_G, pred_scores_P, gt_scores_G, gt_scores_P)
                epoch_loss_Dual = []
                pred_scores_G = []
                pred_scores_P = []
                gt_scores_G = []
                gt_scores_P = []
            '''
            # cuzE
            for t in range(self.epochs):
                try:
                    img, label = self.train_data_cuzE.next()
                except StopIteration:
                    self.train_data_cuzE = iter(self.train_loader_cuzE.get_data())
                    img, label = self.train_data_cuzE.next()
                predP, loss = self.trainOnce(img, label)
                pred_scoresE += predP.detach().cpu().numpy().tolist()
                gt_scoresE = gt_scoresE + label.cpu().tolist()
                epoch_lossE.append(loss.item())

            if (bcnt != 0) and (bcnt % 100 == 0):
                self.analysis(pred_scoresE, gt_scoresE)
                epoch_lossE = []
                pred_scoresE = []
                gt_scoresE = []

            # cuzEG & cuzEP
            '''
             for t in range(self.epochs):
                try:
                    imgG, labelG = self.train_data_cuzEG.next()
                except StopIteration:
                    self.train_data_cuzEG = iter(self.train_loader_cuzEG.get_data())
                    imgG, labelG = self.train_data_cuzEG.next()
                try:
                    imgP, labelP = self.train_data_cuzEP.next()
                except StopIteration:
                    self.train_data_cuzEP = iter(self.train_loader_cuzEP.get_data())
                    imgP, labelP = self.train_data_cuzEP.next()

                predP_G, predP_P, loss = self.trainOnceDual(imgG, labelG, imgP, labelP, 0.25)

                pred_scores_GE += predP_G.detach().cpu().numpy().tolist()
                pred_scores_PE += predP_P.detach().cpu().numpy().tolist()

                gt_scores_GE = gt_scores_GE + labelG.cpu().tolist()
                gt_scores_PE = gt_scores_PE + labelP.cpu().tolist()
                epoch_loss_DualE.append(loss.item())

            if (bcnt != 0) and (bcnt % 100 == 0):
                self.analysisDual(pred_scores_GE, pred_scores_PE, gt_scores_GE, gt_scores_PE)
                epoch_loss_DualE = []
                pred_scores_GE = []
                pred_scores_PE = []
                gt_scores_GE = []
                gt_scores_PE = []

            '''
            if bcnt == 500:
                print('Apply Equal Learning Ratio...')
                lr = self.lr
                self.lrratio = 1
                self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                              {'params': self.model_hyper.res.parameters(), 'lr': lr}
                              ]
                self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

            if (bcnt != 0) and (bcnt % 100 == 0):
                self.test(self.testFR_data, bcnt)
                # torch.save(self.model_hyper, 'Bmodel%d.pkl' % bcnt)
        return 0.0, 0.0

    def test(self, data, t):
        """Testing"""
        self.model_hyper.train(False)
        self.model_hyper.eval()
        with torch.no_grad():
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
            scio.savemat('Cgt%d.mat' % t, {'gt': gt_scores})
            scio.savemat('Cpred%d.mat' % t, {'pred': pred_scores})
            print('Round:%d\t\tPLCC:%.4f\t\tSRCC:%.4f' % (t, test_plcc, test_srcc))
        self.model_hyper.train(True)

        return test_srcc, test_plcc

    def testFR(self, data, t):
        """Testing"""
        self.model_hyper.train(False)
        self.model_hyper.eval()
        with torch.no_grad():
            pred_scores = []
            gt_scores = []
            cnt = 0
            for img, label in data:
                # Data.
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda())
                pred = self.model_hyper(img)
                predP = pred['P']
                pred_scores += predP.detach().cpu().numpy().tolist()
                gt_scores += label.cpu().numpy().tolist()
                cnt += 1

            srcc_15 = []
            for xx in range(16):
                pred = np.array(pred_scores)[:, xx]
                gt = np.array(gt_scores)[:, xx]
                srcc_15.append(stats.spearmanr(pred, gt)[0])
            print(
                'Round:%d\t\tTest:\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' % (
                    t,srcc_15[0], srcc_15[1], srcc_15[2], srcc_15[3], srcc_15[4], srcc_15[5], srcc_15[6], srcc_15[7],
                    srcc_15[8], srcc_15[9], srcc_15[10], srcc_15[11], srcc_15[12], srcc_15[13], srcc_15[14],
                    srcc_15[15]))

            scio.savemat('Cgt%d.mat' % t, {'gt': gt_scores})
            scio.savemat('Cpred%d.mat' % t, {'pred': pred_scores})
        self.model_hyper.train(True)

        return 0.0, 0.0
