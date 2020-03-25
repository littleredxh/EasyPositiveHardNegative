import os, time

from torchvision import models, transforms, datasets
from torch.utils.data.sampler import SequentialSampler
import torch.optim as optim
import torch.nn as nn
import torch

from .Sampler import BalanceSampler, BalanceSampler2
from .Reader import ImageReader
from .Loss import EPHNLoss
from .Utils import recall, recall2, recall2_batch, eva
from .color_lib import RGBmean, RGBstdv

from torch.utils.tensorboard import SummaryWriter

PHASE = ['tra','val']

class learn():
    def __init__(self, dst, Data, data_dict):
        self.dst = dst
        self.gpuid = [0]
            
        self.imgsize = 224
        self.batch_size = 128
        self.num_workers = 32
        
        self.decay_time = [False,False]
        self.init_lr = 0.001
        self.decay_rate = 0.1
        self.avg = 8
        
        self.Data = Data
        self.data_dict = data_dict
        
        self.RGBmean = RGBmean[Data]
        self.RGBstdv = RGBstdv[Data]
        
        self.criterion = EPHNLoss() 
        self.Graph_size = 16
        self.test_freq = 5
        
        self.writer = SummaryWriter(dst)
        self.global_it = 0
        self.multi_gpu = False
        if not self.setsys(): print('system error'); return
        
    def run(self, emb_dim, model_name, num_epochs=20):
        self.out_dim = emb_dim
        self.num_epochs = num_epochs
        self.loadData()
        self.setModel(model_name)
        print('output dimension: {}'.format(emb_dim))
        self.opt()

    ##################################################
    # step 0: System check
    ##################################################
    def setsys(self):
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        return True
    
    ##################################################
    # step 1: Loading Data
    ##################################################
    def loadData(self):
        self.tra_transforms = transforms.Compose([transforms.Resize(int(self.imgsize*1.1)),
                                                  transforms.RandomCrop(self.imgsize),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(self.RGBmean, self.RGBstdv)])
        
        self.val_transforms = transforms.Compose([transforms.Resize(self.imgsize),
                                                  transforms.CenterCrop(self.imgsize),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(self.RGBmean, self.RGBstdv)])

        self.dsets = ImageReader(self.data_dict['tra'], self.tra_transforms) 
        self.intervals = self.dsets.intervals
        self.classSize = len(self.intervals)
        print('number of classes: {}'.format(self.classSize))

        return
    
    ##################################################
    # step 2: Set Model
    ##################################################
    def setModel(self, model_name):
        if model_name == 'R18':
            self.model = models.resnet18(pretrained=True)
            print('Setting model: resnet18')
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.out_dim)
            self.model.avgpool = nn.AvgPool2d(self.avg)
        elif model_name == 'R50':
            self.model = models.resnet50(pretrained=True)
            print('Setting model: resnet50')
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.out_dim)
            self.model.avgpool = nn.AvgPool2d(self.avg)
        elif model_name == 'GBN':
            self.model = models.googlenet(pretrained=True, aux_logits=False, transform_input=False)
            self.model.aux_logits=False
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.classSize)
            print('Setting model: GoogleNet')
            
        else:
            print('model is not exited!')
            
        print('Training on Single-GPU')
        print('LR is set to {}'.format(self.init_lr))
        self.model = self.model.cuda()
        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0,1,2,3], output_device=0)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.init_lr, momentum=0.0)
        return
    
    def lr_scheduler(self, epoch):
        if epoch>=0.5*self.num_epochs and not self.decay_time[0]: 
            self.decay_time[0] = True
            lr = self.init_lr*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch>=0.75*self.num_epochs and not self.decay_time[1]: 
            self.decay_time[1] = True
            lr = self.init_lr*self.decay_rate*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return
            
    ##################################################
    # step 3: Learning
    ##################################################
    def opt(self):
        if self.Data in ['SOP','ICR']:
            batch_limit = 120
        else:
            batch_limit = 30
            
        # recording time
        since = time.time()
    
        for epoch in range(self.num_epochs+1): 
            # adjust the learning rate
            print('Epoch {}/{} \n '.format(epoch, self.num_epochs) + '-' * 40)
            self.lr_scheduler(epoch)
            
            if epoch>0:
                # train 
                tra_loss, N_sample = self.tra()
                while N_sample<self.batch_size*batch_limit:
                    tra_loss_tmp, N_sample_tmp = self.tra(N_limit=self.batch_size*batch_limit-N_sample)
                    tra_loss+=tra_loss_tmp
                    N_sample+=N_sample_tmp
                    
                self.writer.add_scalar('loss', tra_loss/N_sample, epoch)
                
            if epoch%self.test_freq==0:
                # calculate the retrieval accuracy
                if self.Data in ['SOP','CUB','CAR']:
                    acc = self.recall_val2val(epoch)
                elif self.Data=='ICR':
                    acc = self.recall_val2gal(epoch)
                elif self.Data=='HOTEL':
                    acc = self.recall_val2tra(epoch)
                else:
                    acc = self.recall_val2tra(epoch)
                    
                self.writer.add_scalar(self.Data+'_train_R@1', acc[0], epoch)
                self.writer.add_scalar(self.Data+'_test_R@1', acc[1], epoch)

        # save model
        torch.save(self.model.module.cpu(), self.dst + 'model.pth')
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        return
    
    def tra(self, N_limit=-1):
        if self.multi_gpu:
            self.model.module.train(True)  # Set model to training mode with frozen BN
        else:
            self.model.train(True)  # Set model to training mode with frozen BN
            
        if self.Data in ['CUB','CAR']:
            dataLoader = torch.utils.data.DataLoader(self.dsets, batch_size=self.batch_size, sampler=BalanceSampler(self.intervals, GSize=self.Graph_size), num_workers=self.num_workers)
        else: 
            dataLoader = torch.utils.data.DataLoader(self.dsets, batch_size=self.batch_size, sampler=BalanceSampler2(self.intervals, GSize=self.Graph_size), num_workers=self.num_workers)
        
        L_data, N_data = 0.0, 0
        Pos_data, Neg_data = list(), list()
        # iterate batch
        for data in dataLoader:
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
                fvec = self.model(inputs_bt.cuda())
                loss, Pos_log, Neg_log, margin = self.criterion(fvec, labels_bt.cuda())

                loss.backward()
                self.optimizer.step() 
                self.writer.add_histogram('Pos_hist', Pos_log, self.global_it)
                self.writer.add_histogram('Neg_hist', Neg_log, self.global_it)
                self.writer.add_scalar('Margin', margin, self.global_it)
                self.global_it+=1
                
            L_data += loss.item()
            N_data += len(labels_bt)
            if N_data>N_limit and N_limit!=-1: break

        return L_data,N_data
        
    def recall_val2val(self, epoch):
        if self.multi_gpu:
            self.model.module.eval()  # Set model to testing mode
        else:
            self.model.eval()  # Set model to testing mode
        dsets_tra = ImageReader(self.data_dict['tra'], self.val_transforms) 
        dsets_val = ImageReader(self.data_dict['val'], self.val_transforms) 
        Fvec_tra = eva(dsets_tra, self.model)
        Fvec_val = eva(dsets_val, self.model)
        
        if epoch==self.num_epochs-1:
            torch.save(Fvec_tra, self.dst + str(epoch) + 'traFvecs.pth')
            torch.save(Fvec_val, self.dst + str(epoch) + 'valFvecs.pth')
            torch.save(dsets_tra, self.dst + 'tradsets.pth')
            torch.save(dsets_val, self.dst + 'valdsets.pth')
            
        acc_tra = recall(Fvec_tra, dsets_tra.idx_to_class)
        acc_val = recall(Fvec_val, dsets_val.idx_to_class)
        print('R@1_tra:{:.1f}  R@1_val:{:.1f}'.format(acc_tra*100, acc_val*100)) 
        
        return [acc_tra, acc_val]
    
    def recall_val2tra(self, epoch):
        self.model.eval()  # Set model to testing mode
        dsets_tra = ImageReader(self.data_dict['tra'], self.val_transforms) 
        dsets_val = ImageReader(self.data_dict['val'], self.val_transforms) 
        Fvec_tra = eva(dsets_tra, self.model)
        Fvec_val = eva(dsets_val, self.model)

        if epoch==self.num_epochs-1:
            torch.save(Fvec_tra, self.dst + 'traFvecs.pth')
            torch.save(Fvec_val, self.dst + 'valFvecs.pth')
            torch.save(dsets_tra, self.dst + 'tradsets.pth')
            torch.save(dsets_val, self.dst + 'valdsets.pth')

        acc = recall2(Fvec_val, Fvec_tra, dsets_val.idx_to_class, dsets_tra.idx_to_class)
        print('R@1:{:.2f}'.format(acc)) 
        
        return [acc]
    
    def recall_val2gal(self, epoch):
        self.model.eval()  # Set model to testing mode
        dsets_gal = ImageReader(self.data_dict['gal'], self.val_transforms) 
        dsets_val = ImageReader(self.data_dict['val'], self.val_transforms) 
        Fvec_gal = eva(dsets_gal, self.model)
        Fvec_val = eva(dsets_val, self.model)
        
        if epoch==self.num_epochs-1:
            torch.save(Fvec_gal, self.dst + 'galFvecs.pth')
            torch.save(Fvec_val, self.dst + 'valFvecs.pth')
            torch.save(dsets_gal, self.dst + 'galdsets.pth')
            torch.save(dsets_val, self.dst + 'valdsets.pth')
            
        acc = recall2(Fvec_val, Fvec_gal, dsets_val.idx_to_class, dsets_gal.idx_to_class)
        print('R@1:{:.2f}'.format(acc)) 
        
        return [acc,acc]
    