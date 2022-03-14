import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

# PyTorch classifier
class PtImageClassifier(nn.Module):
    def __init__(self, channel, height, width, num_classes):
        super().__init__()
        # Model layers
        self.layer1 = nn.Linear(in_features=channel*width*height, 
                                out_features=256)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.25)
        self.layer2 = nn.Linear(in_features=256, 
                                out_features=64)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.25)
        self.layer3 = nn.Linear(in_features=64, 
                                out_features=num_classes)
    
            
    def forward(self, x):
        # flatten the input
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.do1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.do2(x)
        out = self.layer3(x)
        return out


# Lightning wrapper 
class ImageClassifier(pl.LightningModule):
    def __init__(self, channel, height, width, num_classes):
        super().__init__()
        # Save passed hyper-parameters to check-point
        self.save_hyperparameters()
        # Metrics
        self.metrics_ = {'train': {
                                  'batch': {'loss':[], 'acc':[]}, 
                                  'epoch':{'loss':[], 'acc':[]}
                                 }, 
                         'val': {
                                  'batch': {'loss':[], 'acc':[]}, 
                                  'epoch':{'loss':[], 'acc':[]}
                                }, 
                         'test': {
                                  'batch': {'loss':[], 'acc':[]}, 
                                  'epoch':{'loss':[], 'acc':[]}
                                }                          
                        }
        # Track epoch and batches for metrics calculations
        self.__curr_epoch = 0
        self.__n_train_batches = 0
        self.__n_val_batches = 0
        self.__n_test_batches = 0
        # Loss function
        self.criterion = nn.CrossEntropyLoss()        
        # Instantiate main model
        self.model = PtImageClassifier(channel, height, width, num_classes)

    
    # Private method to performed forward pass and calculate
    # loss and accuracy. Called from train, val and test loops
    def __step(self, batch):
        x, y = batch
        # Get model output as raw logits
        logits = self.model(x) # calls forward of the main model
        # Compute loss and accuracy
        loss = self.criterion(logits, y)
        acc = accuracy(F.softmax(logits, dim=-1), y)   
        return loss, acc
 
 
    # Private method to calculate epoch level metrics from batch
    # level metrics.Called from train, val and test loops        
    def __compute_epoch_metric(self, s=None, m=None, n=0, e=0):
        start_idx = e * n + 1
        end_idx = start_idx + n + 1
        batch_m = self.metrics_[s]['batch'][m][start_idx:end_idx]
        epoch_m = sum(batch_m)/len(batch_m)
        return epoch_m

    # Called by lightning at the start of each training batch
    def on_training_start(self):
        sys.stdout.flush()
        print()


    # Called by lightning to process a training batch 
    def training_step(self, batch, batch_idx):
        self.__n_train_batches = batch_idx
        # Feed forward and get loss and metrics
        loss, acc = self.__step(batch)
        # store batch loss and accuracy
        l = loss.detach().item()
        a = acc.detach().item()
        self.metrics_['train']['batch']['loss'].append(l)
        self.metrics_['train']['batch']['acc'].append(a)
        # print the running status
        sys.stdout.write(f'\rTraining Epoch   [{self.__curr_epoch+1:4d}] ==> train_loss: {l:.8f}, train_acc: {a*100:.2f}% ({batch_idx})')        
        return {'loss': loss}

    # Called by lightning at the start of each validation batch
    def on_validation_start(self):
        sys.stdout.flush()
        print()


    # Called by lightning to process a validation batch 
    def validation_step(self, batch, batch_idx):
        self.__n_val_batches = batch_idx
        # Feed forward and get loss and metrics        
        loss, acc = self.__step(batch)
        # store batch loss and accuracy
        l = loss.detach().item()
        a = acc.detach().item()
        self.metrics_['val']['batch']['loss'].append(l)
        self.metrics_['val']['batch']['acc'].append(a)
        # print the running status
        sys.stdout.write(f'\rValidating Epoch [{self.__curr_epoch+1:4d}] ==> val_loss  : {l:.8f}, val_acc  : {a*100:.2f}% ({batch_idx})')
        return {'loss': loss}


    # Called by lightning at the end of training epoch 
    def training_epoch_end(self, outputs):
        print()
        # compute this epoch's metrics
        # get batch level metrics for this epoch and compute average
        # train 
        loss = self.__compute_epoch_metric('train', 'loss', 
                                            self.__n_train_batches,
                                            self.__curr_epoch)
        self.metrics_['train']['epoch']['loss'].append(loss)
        acc = self.__compute_epoch_metric('train', 'acc', 
                                          self.__n_train_batches,
                                          self.__curr_epoch)
        self.metrics_['train']['epoch']['acc'].append(acc)

        # val 
        loss = self.__compute_epoch_metric('val', 'loss', 
                                            self.__n_val_batches,
                                            self.__curr_epoch)
        self.metrics_['val']['epoch']['loss'].append(loss)
        acc = self.__compute_epoch_metric('val', 'acc', 
                                            self.__n_val_batches,
                                            self.__curr_epoch)
        self.metrics_['val']['epoch']['acc'].append(acc)

        # prepare for next epoch
        self.__curr_epoch += 1
        self.__n_train_batches = 0
        self.__n_val_batches = 0


    # Called by lightning when Trained.test is executed 
    def test_step(self, batch, batch_idx):
        self.__n_test_batches = batch_idx
        # Feed forward and get loss and metrics
        loss, acc = self.__step(batch)
        # store batch loss and accuracy
        l = loss.detach().item()
        a = acc.detach().item()
        self.metrics_['test']['batch']['loss'].append(l)
        self.metrics_['test']['batch']['acc'].append(a)
        # print the running status
        sys.stdout.write(f'\rTesting ==> test_acc: {a*100:.2f}% ({batch_idx})')        
        return {'loss': l, 'acc':a}

    # Called by lightning at the end of test  
    def test_epoch_end(self, outputs):
        print()
        # compute this epoch's metrics
        # get batch level metrics for this epoch and compute average
        # train 
        loss = self.__compute_epoch_metric('test', 'loss', 
                                            self.__n_test_batches,
                                            0)
        self.metrics_['test']['epoch']['loss'].append(loss)
        acc = self.__compute_epoch_metric('test', 'acc', 
                                          self.__n_test_batches,
                                          0)
        self.metrics_['test']['epoch']['acc'].append(acc)
        # Some how log is required to print when test ends
        test_result = {'loss':loss, 'acc': acc}
        self.log_dict(test_result)


    # Define Optimizers
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-2)

    # Return requested metrics as a DataFrame
    def metrics(self, stage='train', level='batch'):
        return pd.DataFrame(self.metrics_[stage][level])
        
    # This method is called by Trainer.predict()
    def predict_step(self, batch, batch_idx):
        x = batch[0]
        # Get prediction from as raw logits from main model
        logits = self.model(x) 
        # Find the index of max probability
        pred_proba = F.softmax(logits, dim=-1)
        preds = torch.argmax(pred_proba, dim=-1)   
        return {'preds':preds, 'pred_proba':pred_proba}