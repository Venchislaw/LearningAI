import numpy as np


class Loss:
    def calculate(self, y_true, y_pred):
        self.sample_losses = self.forward(y_true, y_pred)
        general_data_loss = np.mean(self.sample_losses)

        return general_data_loss
    

class CategoricalCrossentropy(Loss):
    def forward(self, y_true, y_pred):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            confidences = y_pred_clipped[range(len(y_true)), y_true]
        
        elif len(y_true.shape) == 2:
            confidences = np.sum(y_pred_clipped * y_true, axis=1)

        loss = -np.log(confidences)

        return loss

