import os, time, torch
import argparse

from torch.utils.tensorboard import SummaryWriter

from _code.color_lib import RGBmean, RGBstdv
from _code.Loss import EPHNLoss
from _code.Model import setModel, setOptimizer
from _code.Sampler import BalanceSampler_sample, BalanceSampler_filled
from _code.Reader import ImageReader
from _code.Utils import tra_transforms, eva_transforms
from _code.Evaluation import test

parser = argparse.ArgumentParser(description='running parameters')
parser.add_argument('--Data', type=str, help='dataset name: CUB, CAR, SOP or ICR')
parser.add_argument('--model', type=str, help='backbone model: R18 or R50')
parser.add_argument('--dim', type=int, help='embedding dimension')
parser.add_argument('--lr', type=float, help='initial learning rate')
parser.add_argument('--method', type=str, help='order')
parser.add_argument('--nsize', type=int, help='nsize')
parser.add_argument('--epochs', type=int, help='epochs')
args = parser.parse_args()


# data dict
Data = args.Data
dst = '_result/{}/{}_{}/G{}/'.format(args.method, args.Data, args.model, args.nsize)
data_dict = torch.load('/home/xuanhong/datasets/{}/data_dict_emb.pth'.format(Data))
phase = ['tra','val']

# dataset setting
imgsize = 256
tra_transform = tra_transforms(imgsize, RGBmean[Data], RGBstdv[Data])
eva_transform = eva_transforms(imgsize, RGBmean[Data], RGBstdv[Data])

# network setting
model_name = args.model
emb_dim = args.dim
multi_gpu = False

# sampler setting
batch_size = 128
num_workers = 32
print('batch size: {}'.format(batch_size))

# loss setting
criterion = EPHNLoss() 
N_size = args.nsize
print('number of images per class: {}'.format(N_size))

# recorder frequency
num_epochs = args.epochs
test_freq = 5
writer = SummaryWriter(dst)


# model setting
model = setModel(model_name, emb_dim).cuda()
print('output dimension: {}'.format(emb_dim))

if multi_gpu:
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3], output_device=0)

# Optimizer and scheduler setting
optimizer, scheduler = setOptimizer(model.parameters(), args.lr, [int(num_epochs*0.5), int(num_epochs*0.75)])

        
# training
since = time.time() # recording time
global_it = 0
for epoch in range(num_epochs+1): 

    print('Epoch {}/{} \n '.format(epoch, num_epochs) + '-' * 40)

    # train phase
    if epoch>0:  
        # create dset
        dsets = ImageReader(data_dict['tra'], tra_transform) 

        # create sampler
        if Data in ['SOP','ICR']:
            sampler = BalanceSampler_sample(dsets.intervals, GSize=N_size)
        else:
            sampler = BalanceSampler_filled(dsets.intervals, GSize=N_size)

        # create dataloader
        dataLoader = torch.utils.data.DataLoader(dsets, batch_size=batch_size, sampler=sampler, num_workers=32)
        
        # Set model to training mode
        if multi_gpu:
            model.module.train()
        else:
            model.eval()
 
        # record loss
        L_data, N_data = 0.0, 0

        # iterate batch
        for data in dataLoader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
                fvec = model(inputs_bt.cuda())
                loss, Pos_log, Neg_log, margin = criterion(fvec, labels_bt.cuda())
                loss.backward()
                optimizer.step() 
            writer.add_histogram('Pos_hist', Pos_log, global_it)
            writer.add_histogram('Neg_hist', Neg_log, global_it)
            writer.add_scalar('Margin', margin, global_it)
            global_it+=1

            L_data += loss.item()
            N_data += len(labels_bt)
            
        writer.add_scalar('loss', L_data/N_data, epoch)
        # adjust the learning rate
        scheduler.step()
    
    # evaluation phase
    if epoch%test_freq==0:
        # evaluate train set
        dsets_dict = {p: ImageReader(data_dict[p], eva_transform) for p in phase}
        acc = test(Data, dsets_dict, model, epoch, writer, multi_gpu=False)
        
    
    
# save model
torch.save(model.cpu().state_dict(), dst + 'model_state_dict.pth')
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))


