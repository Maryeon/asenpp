import math
import random

class MetricScorer:

    def __init__(self, k=0):
        self.k = k

    def score(self, sorted_labels):
        return 0.0

    def getLength(self, sorted_labels):
        length = self.k
        if length>len(sorted_labels) or length<=0:
            length = len(sorted_labels)
        return length    

    def name(self):
        if self.k > 0:
            return "%s@%d" % (self.__class__.__name__.replace("Scorer",""), self.k)
        return self.__class__.__name__.replace("Scorer","")

    def setLength(self, k):
        self.k = k;


class APScorer (MetricScorer):
 
    def __init__(self, k=0):
        MetricScorer.__init__(self, k)
        

    def score(self, sorted_labels):
        nr_relevant = len([x for x in sorted_labels if x > 0])
        if nr_relevant == 0:
            return 0.0
            
        length = self.getLength(sorted_labels)
        ap = 0.0
        rel = 0
        
        for i in range(length):
            lab = sorted_labels[i]
            if lab >= 1:
                rel += 1
                ap += float(rel) / (i+1.0)
        ap /= nr_relevant
        return ap

# reciprocal rank
class RRScorer (MetricScorer):

    def score(self, sorted_labels):
        for i in range(len(sorted_labels)):
            if 1 <= sorted_labels[i]:
                return 1.0/(i+1)
        return 0.0


class PrecisionScorer (MetricScorer):

    def score(self, sorted_labels):
        length = self.getLength(sorted_labels)

        rel = 0
        for i in range(length):
            if sorted_labels[i] >= 1:
                rel += 1

        return float(rel)/length

    
class NDCGScorer (PrecisionScorer):

    def score(self, sorted_labels):
        d = self.getDCG(sorted_labels)
        d2 = self.getIdealDCG(sorted_labels) 
        return d/d2

    def getDCG(self, sorted_labels):
        length = self.getLength(sorted_labels)

        dcg = max(sorted_labels[0], 0)
        #print dcg
        for i in range(1, length):
            rel = max(sorted_labels[i], 0)
            dcg += float(rel)/math.log(i+1, 2)
            #print i, sorted_labels[i], math.log(i+1,2), float(sorted_labels[i])/math.log(i+1, 2)
        return dcg

    def getIdealDCG(self, sorted_labels):
        ideal_labels = sorted(sorted_labels, reverse=True)
        return self.getDCG(ideal_labels)


class DCGScorer (PrecisionScorer):

    def score(self, sorted_labels):
        return self.getDCG(sorted_labels)

    def getIdealDCG(self, sorted_labels):
        ideal_labels = sorted(sorted_labels, reverse=True)
        return self.getDCG(ideal_labels)

    def getRandomDCG(self, sorted_labels):
        random.shuffle(sorted_labels)
        return self.getDCG(sorted_labels)

    def getDCG(self, sorted_labels):
        # while(self.getLength(sorted_labels) < self.k) : sorted_labels.append(0)
        dcgPart = [(math.pow( 2, rel)-1)/ math.log(index+1, 2) for index, rel in enumerate(sorted_labels[:self.k],1)]
        # dcgPart = [(math.pow( 2, rel)-1)/ math.log(index+1, 2) for index, rel in enumerate(sorted_labels[:25],1)]        
        return 0.01757*sum(dcgPart)


def getScorer(name):
    mapping = {"P":PrecisionScorer, "AP":APScorer, "RR":RRScorer,  "NDCG":NDCGScorer, "DCG":DCGScorer}
    elems = name.split("@")
    if len(elems) == 2:
        k = int(elems[1])
    else:
        k = 0
    return mapping[elems[0]](k)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count