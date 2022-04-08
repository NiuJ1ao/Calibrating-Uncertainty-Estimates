# import numpy as np
import torch
      
# the difference in expectation between confidence and accuracy
def expected_calibration_error(predictions, confidences, references, device="cpu"):
    n = len(references)
    _, bin_sizes, bin_accs, bin_avg_confs = partition_bins(predictions, confidences, references, device)
    assert bin_sizes.shape == bin_accs.shape == bin_avg_confs.shape, (bin_sizes.shape, bin_accs.shape, bin_avg_confs.shape)
    
    ece = torch.sum(bin_sizes * torch.abs(bin_accs - bin_avg_confs) / n)
    return ece

def multiclass_ece(predictions, confidences, references):
    eces = torch.tensor([])
    for c, confs in enumerate(confidences.T):
        ece = expected_calibration_error(predictions, confs, references)
        eces = torch.cat((eces, ece.unsqueeze(0)))
    return eces.mean()

'''
Auxiliary functions
'''
def partition_bins(predictions, confidences, references, device="cpu"):
    bin_size = 0.1
    num_bins = int(1 / bin_size)
    bins = torch.linspace(bin_size, 1, num_bins, device=device)
    bin_inds = torch.bucketize(confidences, bins)
    
    if len(bin_inds) < num_bins:
        print("    WARNING: Some bins are zero. Try to increase bin size.")
    
    bin_accs = torch.tensor([], device=device)
    bin_avg_confs = torch.tensor([], device=device)
    bin_sizes = torch.tensor([], device=device)
    for bin in range(num_bins):
        bin_size = torch.count_nonzero(bin_inds == bin)
        
        bin_refs = references[bin_inds==bin]
        bin_preds = predictions[bin_inds==bin]
        bin_confs = confidences[bin_inds==bin]
        
        acc = accuracy(bin_preds, bin_refs, device)
        conf = average_confidence(bin_confs, device)
        
        bin_sizes = torch.cat((bin_sizes, bin_size.unsqueeze(0)))
        bin_accs = torch.cat((bin_accs, acc.unsqueeze(0)))
        bin_avg_confs = torch.cat((bin_avg_confs, conf.unsqueeze(0)))
        
    return bins, bin_sizes, bin_accs, bin_avg_confs
    
def accuracy(predictions, references, device="cpu"):
    if len(predictions) == 0:
        return torch.tensor(0., device=device)
    acc = torch.count_nonzero(predictions == references) / len(predictions)
    return acc

def average_confidence(confidences, device="cpu"):
    if len(confidences) == 0:
        return torch.tensor(1., device=device)
    conf = torch.mean(confidences)
    return conf
    

if __name__ == "__main__":
    preds = torch.tensor([1,0,1])
    refs = torch.tensor([1,0,2])
    confs = torch.tensor([[0.7, 0.3, 0.8], [0.1, 0.2, 0.9], [0.4, 0.5, 0.6]])
    print(multiclass_ece(preds, confs, refs))
    # print(partition_bins(preds, confs, refs))
    # print(expected_calibration_error(preds, confs, refs))