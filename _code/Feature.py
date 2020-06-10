import torch
from torch.utils.data.sampler import SequentialSampler
import torch.nn.functional as F

def feature(dsets, model):
    Fvecs = []
    dataLoader = torch.utils.data.DataLoader(dsets, batch_size=400, sampler=SequentialSampler(dsets), num_workers=48)
    torch.set_grad_enabled(False)
    model.eval()
    for data in dataLoader:
        inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
        fvec = model(inputs_bt.cuda())
        fvec = F.normalize(fvec, p = 2, dim = 1).cpu()
        Fvecs.append(fvec)
            
    return torch.cat(Fvecs,0)