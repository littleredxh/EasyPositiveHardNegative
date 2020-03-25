from _code.Train import learn
import os, torch
import argparse

parser = argparse.ArgumentParser(description='running parameters')
parser.add_argument('--Data', type=str, help='dataset name: CUB, CAR, SOP or ICR')
parser.add_argument('--model', type=str, help='backbone model: R18 or R50')
parser.add_argument('--dim', type=int, help='embedding dimension')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--method', type=str, help='order')
parser.add_argument('--g', type=int, help='gsize')
parser.add_argument('--ep', type=int, help='epochs')
args = parser.parse_args()

data_dict = torch.load('/home/xuanhong/datasets/{}/data_dict_emb.pth'.format(args.Data))
dst = '_result/{}/{}_{}/G{}/'.format(args.method,args.Data,args.model,args.g)
print(dst)

x = learn(dst, args.Data, data_dict)
x.batch_size = 128
x.Graph_size = args.g
x.init_lr = args.lr
if args.method=='EPSHN':
    x.criterion.semi = True
x.run(args.dim, args.model, num_epochs=args.ep)

# SOP_EPHN = ['SOP','R50',512,0.001,'EPHN',0.1]
# ICR_EPHN = ['ICR','R50',512,0.001,'EPHN',0.1]
# CUB_EPHN = ['CUB','R50',512,0.001,'EPHN',0.1]
# CAR_EPHN = ['CAR','R50',512,0.001,'EPHN',0.1]

# SOP_EPSHN = ['SOP','R50',512,0.001,'EPSHN',0.1]
# ICR_EPSHN = ['ICR','R50',512,0.001,'EPSHN',0.1]
# CUB_EPSHN = ['CUB','R50',512,0.001,'EPSHN',0.1]
# CAR_EPSHN = ['CAR','R50',512,0.001,'EPSHN',0.1]
