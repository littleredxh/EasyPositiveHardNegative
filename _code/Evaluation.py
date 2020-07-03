import torch

from .Feature import feature
from _code.Utils import recall, recall2, recall2_batch

def test_hotel(dst, dsets_dict, model, epoch, writer, multi_gpu=False):
    
    if multi_gpu:
        model.module.eval()  # Set model to testing mode
    else:
        model.eval()  # Set model to testing mode
        
    # train set feature
    Fvec_tra = feature(dsets_dict['tra'], model)
    torch.save(Fvec_tra, dst + str(epoch) + 'traFvecs.pth')
        
    # test set feature
    Fvec_test = eva(dsets_dict['test'], model)
    torch.save(Fvec_test, dst + str(epoch) + 'testFvecs.pth')

    # top100 NN index
    _, pre100 = recall2_batch(Fvec_test, Fvec_tra, dsets_dict['test'].idx_to_class, dsets_dict['tra'].idx_to_class)
    torch.save(pre100, dst+str(epoch)+'pre100.pth')
    
    return
        
        
def test(Data, dsets_dict, model, epoch, writer, multi_gpu=False):
    
    if multi_gpu:
        model.module.eval()  # Set model to testing mode
    else:
        model.eval()  # Set model to testing mode
        
    # calculate the retrieval accuracy
    if Data=='ICR':
        # test set r@1
        acc = recall2(feature(dsets_dict['test'], model),
                      feature(dsets_dict['gal'], model), 
                      dsets_dict['test'].idx_to_class, 
                      dsets_dict['gal'].idx_to_class)
        
        print('R@1:{:.2f}'.format(acc)) 
        
        writer.add_scalar(Data+'_test_R@1', acc, epoch)
        
    else:
        # train set r@1
        acc_tra = recall(feature(dsets_dict['tra'], model), dsets_dict['tra'].idx_to_class)
        # test set r@1
        acc_test = recall(feature(dsets_dict['test'], model), dsets_dict['test'].idx_to_class)
        
        print('R@1_tra:{:.1f} R@1_test:{:.1f}'.format(acc_tra*100, acc_test*100)) 
        
        writer.add_scalar(Data+'_train_R@1', acc_tra, epoch)
        writer.add_scalar(Data+'_test_R@1', acc_test, epoch)
        
    return