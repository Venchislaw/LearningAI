import numpy as np


# accuracy for classification:

class CategoricalAccuracy:
    def get_accuracy(self, y_true, y_pred):
        
        if len(y_true.shape) == 1:
            y_preds = np.argmax(y_pred, axis=1)
            return np.sum(y_true == y_preds) / len(y_true)
        
        elif len(y_true.shape) == 2:
            y_preds = np.argmax(y_pred, axis=1)
            y_trues = np.argmax(y_true, axis=1)

            return np.sum(y_trues == y_preds) / len(y_true)
