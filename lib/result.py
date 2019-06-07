from pdb import set_trace

class Result(object):
    def __init__(self, predictions, labels):
        assert len(predictions) == len(labels), "The length of the predictions and the length of the labels do not match!"
        self.predictions = predictions
        self.labels = labels

    def accuracy(self):
        correct = (self.predictions == self.labels).sum().item()
        total = len(self.predictions)

        return (correct * 1.0) / total
