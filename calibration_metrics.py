# import numpy as np
import torch
      
# the difference in expectation between confidence and accuracy
def expected_calibration_error(predictions, confidences, references):
    n = len(predictions)
    _, bin_sizes, bin_accs, bin_avg_confs = partition_bins(predictions, confidences, references)
    
    ece = torch.sum(bin_sizes * torch.abs(bin_accs - bin_avg_confs) / n)
    return ece

'''
Auxiliary functions
'''
def partition_bins(predictions, confidences, references):
    bin_size = 0.1
    num_bins = int(1 / bin_size)
    bins = torch.linspace(bin_size, 1, num_bins)
    bin_inds = torch.bucketize(confidences, bins)
    
    if len(bin_inds) < num_bins:
        print("    WARNING: Some bin size is zero. Try to increase bin size.")
    
    bin_accs = torch.tensor([])
    bin_avg_confs = torch.tensor([])
    bin_sizes = torch.tensor([])
    for bin in range(num_bins):
        bin_size = torch.count_nonzero(bin_inds == bin)
        bin_sizes = torch.cat((bin_sizes, bin_size.unsqueeze(0)))
        
        bin_refs = references[bin_inds==bin]
        bin_preds = predictions[bin_inds==bin]
        bin_confs = confidences[bin_inds==bin]
        
        acc = accuracy(bin_preds, bin_refs)
        bin_accs = torch.cat((bin_accs, acc.unsqueeze(0)))
        conf = average_confidence(bin_confs)
        bin_avg_confs = torch.cat((bin_avg_confs, conf.unsqueeze(0)))
        
    return bins, bin_sizes, bin_accs, bin_avg_confs
    
def accuracy(predictions, references):
    if len(predictions) == 0:
        return torch.tensor(0.)
    acc = torch.count_nonzero(predictions == references) / len(predictions)
    return acc

def average_confidence(confidences):
    if len(confidences) == 0:
        return torch.tensor(0.)
    conf = torch.mean(confidences)
    return conf
    

if __name__ == "__main__":
    preds = torch.tensor([1,0,1])
    refs = torch.tensor([1,1,1])
    confs = torch.tensor([0.7, 0.3, 0.8])
    print(partition_bins(preds, refs, confs))
    print(expected_calibration_error(preds, refs, confs))