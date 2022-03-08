import numpy as np
      
# the difference in expectation between confidence and accuracy
def expected_calibration_error(predictions, references, confidences):
    n = np.sum(predictions.shape)
    _, bin_sizes, bin_accs, bin_avg_confs = partition_bins(predictions, references, confidences)
    
    ece = np.sum(bin_sizes * np.abs(bin_accs - bin_avg_confs) / n)
    return ece

'''
Auxiliary functions
'''
def partition_bins(predictions, references, confidences):
    bin_size = 0.1
    num_bins = int(1 / bin_size)
    bins = np.linspace(bin_size, 1, num_bins)
    bin_inds = np.digitize(confidences, bins)
    
    if len(bin_inds) < num_bins:
        print("    WARNING: Some bin size is zero. Try to increase bin size.")
    
    bin_accs = []
    bin_avg_confs = []
    bin_sizes = []
    for bin in range(num_bins):
        bin_size = np.count_nonzero(bin_inds == bin)
        bin_sizes.append(bin_size)
        
        bin_refs = references[bin_inds==bin]
        bin_preds = predictions[bin_inds==bin]
        bin_confs = confidences[bin_inds==bin]
        
        bin_accs.append(accuracy(bin_preds, bin_refs))
        bin_avg_confs.append(average_confidence(bin_confs))
        
    return bins, np.array(bin_sizes), np.array(bin_accs), np.array(bin_avg_confs)
    
def accuracy(predictions, references):
    if len(predictions) == 0:
        return 0.
    return np.count_nonzero(predictions == references) / len(predictions)

def average_confidence(confidences):
    if len(confidences) == 0:
        return 0.
    return np.mean(confidences)
    

if __name__ == "__main__":
    preds = np.array([1,0,1])
    refs = np.array([1,1,1])
    confs = np.array([0.7, 0.3, 0.8])
    print(partition_bins(preds, refs, confs))
    print(expected_calibration_error(preds, refs, confs))