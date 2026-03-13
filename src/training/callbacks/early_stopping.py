'''
currently, the early stopping criteria is set to validation loss,
TODO: expand the pipeline to accept other losses
'''
from utils.device import is_main_process

class EarlyStopping:
    def __init__(self, patience, delta, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None # updates the best_
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < (self.best_loss - self.delta):
            # criteria to update best_loss: None or new validation loss is smaller
            self.best_loss = val_loss 
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if is_main_process():
                    print("Stopping early as no improvement has been observed.") 
        
