import torch

def singleBinary(preds, labels, mask=None,dim=6):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()

    # Move batch dimension to the end
    preds = preds.flatten(2, -1).permute(1, 0, 2).reshape(dim, -1)
    labels = labels.flatten(2, -1).permute(1, 0, 2).reshape(dim, -1)
    if mask is not None:
        mask = mask.cpu()
        preds = preds[:, mask.flatten()]
        labels = labels[:, mask.flatten()]
        # true_pos2 = preds2 & labels2
        # false_pos2 = preds2 & ~labels2
        # false_neg2 = ~preds2 & labels2
        # tp2 = true_pos2.long().sum(-1)
        # fp2 = false_pos2.long().sum(-1)
        # fn2 = false_neg2.long().sum(-1)
    true_pos = preds & labels
    false_pos = preds & ~labels
    false_neg = ~preds & labels

    # Update global counts
    tp = true_pos.long().sum(-1)
    fp = false_pos.long().sum(-1)
    fn = false_neg.long().sum(-1)
    iou = tp.float() / (tp + fn + fp + 1e-7).float()
    # valid = (tp + fn) > 0
    # if not valid.any():
    #     return iou,0
    miou= float(iou.mean())
    return iou,miou
class BinaryConfusionMatrix(object):

    def __init__(self, num_class):
        self.tp = torch.zeros(num_class, dtype=torch.long)
        self.fp = torch.zeros(num_class, dtype=torch.long)
        self.fn = torch.zeros(num_class, dtype=torch.long)
        self.tn = torch.zeros(num_class, dtype=torch.long)


    @property
    def num_class(self):
        return len(self.tp)
    
    def update(self, preds, labels, mask=None):

        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        # Move batch dimension to the end
        preds = preds.flatten(2, -1).permute(1, 0, 2).reshape(
            self.num_class, -1)
        labels = labels.flatten(2, -1).permute(1, 0, 2).reshape(
            self.num_class, -1)

        if mask is not None:
            preds = preds[:, mask.flatten()]
            labels = labels[:, mask.flatten()]
        

        true_pos = preds & labels
        false_pos = preds & ~labels
        false_neg = ~preds & labels
        true_neg = ~preds & ~labels

        # Update global counts
        self.tp += true_pos.long().sum(-1)
        self.fp += false_pos.long().sum(-1)
        self.fn += false_neg.long().sum(-1)
        self.tn += true_neg.long().sum(-1)
    

    @property
    def iou(self):
        return self.tp.float() / (self.tp + self.fn + self.fp+1e-7).float()
    
    @property
    def mean_iou(self):
        # Only compute mean over classes with at least one ground truth
        # valid = (self.tp + self.fn) > 0
        # if not valid.any():
        #     return 0
        return float(self.iou.mean())

    @property
    def dice(self):
        return 2 * self.tp.float() / (2 * self.tp + self.fp + self.fn).float()
    
    @property
    def macro_dice(self):
        valid = (self.tp + self.fn) > 0
        if not valid.any():
            return 0
        return float(self.dice[valid].mean())
    
    @property
    def precision(self):
        return self.tp.float() / (self.tp + self.fp).float()
    
    @property
    def recall(self):
        return self.tp.float() / (self.tp + self.fn).float()