import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import f1_score, accuracy_score
import json
import os
import numpy as np


# Helper function to save metrics
def save_metrics(metrics, save_dir, filename="metrics.json"):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, filename), 'w') as f:
            json.dump(metrics, f, indent=4)


# Advanced Setup Model with Shared Training/Validation Functions
class AdvancedSetup(nn.Module):
    def __init__(self):
        super(AdvancedSetup, self).__init__()

    def train_loader(self, data_loader, device, num_epochs, criterion, optimizer=None, scheduler=None,
                     savedir=None, f1_score_average='macro', labels=None, validation_mode=False,
                     use_instance_labels=True, use_bag_labels=False, pooling_method='mean',
                     verbose=True, one_epoch_mode=False, return_predictions=False):
        metrics = {'epochs': [], 'loss': [], 'accuracy': [], 'f1': []}
        
        if one_epoch_mode:
            original_num_epochs = num_epochs
            num_epochs = 1
        
        for epoch in range(num_epochs):
            epoch_loss, epoch_accuracy, epoch_f1 = 0.0, 0.0, 0.0
            loop_count = 0
            epoch_bag_predictions = []

            for data in data_loader:
                loop_count += 1
                inputs = data[0]
                batch_num = inputs.shape[0]
                instance_num = inputs.shape[1]
                features_dim = tuple(inputs.shape[2:])
                inputs = inputs.view(batch_num * instance_num, *features_dim)  # Reshape for forward pass

                if use_bag_labels and not use_instance_labels:
                    targets = data[1]
                elif use_instance_labels and not use_bag_labels:
                    targets = data[2].flatten()
                elif use_instance_labels and use_bag_labels:
                    raise ValueError("Choose only one of 'use_instance_labels' or 'use_bag_labels'")

                if targets.dtype != torch.long:
                    targets = targets.long()

                inputs, targets = inputs.to(device), targets.to(device)
                self.to(device)

                if validation_mode:
                    self.eval()
                    with torch.no_grad():
                        outputs = self._process_outputs(inputs, batch_num, instance_num, pooling_method)
                        loss = criterion(outputs, targets)
                else:
                    self.train()
                    outputs = self._process_outputs(inputs, batch_num, instance_num, pooling_method)
                    loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if scheduler:
                        scheduler.step(loss)

                outputs = outputs.cpu()
                targets = targets.cpu()
                bag_predictions = torch.argmax(outputs, dim=1).numpy()
                if return_predictions:
                    epoch_bag_predictions.append(bag_predictions)

                batch_accuracy = accuracy_score(targets, bag_predictions)
                batch_f1 = f1_score(targets, bag_predictions, average=f1_score_average, labels=labels, zero_division=np.nan)
                epoch_loss += loss.item()
                epoch_accuracy += batch_accuracy
                epoch_f1 += batch_f1

            epoch_loss /= loop_count
            epoch_accuracy /= loop_count
            epoch_f1 /= loop_count

            if one_epoch_mode:
                metrics['epochs'].append(original_num_epochs)
            else:
                metrics['epochs'].append(epoch)
            metrics['loss'].append(epoch_loss)
            metrics['accuracy'].append(epoch_accuracy)
            metrics['f1'].append(epoch_f1)

            if verbose:
                mode_text = 'VALIDATE' if validation_mode else 'TRAINING'
                print(f"{mode_text} | Epoch {epoch + 1:03d} | Loss: {epoch_loss:.6f} | Accuracy: {epoch_accuracy:.4f} | F1: {epoch_f1:.4f}")

            if savedir:
                stat_filename = 'stats_val.json' if validation_mode else 'stats_train.json'
                with open(os.path.join(savedir, stat_filename), 'w') as json_file:
                    json.dump(metrics, json_file, indent=4)

        if return_predictions:
            return self, metrics, epoch_bag_predictions
        else:
            return self, metrics

    def val_loader(self, val_loader, device, criterion, f1_average, labels, pooling_method, verbose):
        return self.train_loader(
            val_loader, device, num_epochs=1, criterion=criterion, validation_mode=True,
            pooling_method=pooling_method, verbose=verbose, use_instance_labels=False, use_bag_labels=True,
            return_predictions=False, f1_score_average=f1_average, labels=labels)

    def train_and_validate(self, train_loader, val_loader, device, num_epochs, criterion, optimizer,
                           scheduler=None, f1_average='macro', labels=None, pooling_method='mean', verbose=True,
                           savedir=None, checkpoint_epochs=[], save_model=True):
        main_metrics = {'epochs': [], 'train_loss': [], 'train_accuracy': [], 'train_f1': [],
                        'val_loss': [], 'val_accuracy': [], 'val_f1': []}

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_loader(
                train_loader, device, num_epochs=1, criterion=criterion, optimizer=optimizer,
                scheduler=scheduler, f1_score_average=f1_average, labels=labels, pooling_method=pooling_method, verbose=verbose)
            val_metrics = self.val_loader(
                val_loader, device, criterion, f1_average, labels, pooling_method, verbose=verbose)

            main_metrics['epochs'].append(epoch)
            for key in ['loss', 'accuracy', 'f1']:
                main_metrics[f'train_{key}'].append(train_metrics[1][key][-1])
                main_metrics[f'val_{key}'].append(val_metrics[1][key][-1])

            save_metrics(main_metrics, savedir, filename=f'stats_epoch_{epoch}.json')

            if epoch in checkpoint_epochs:
                checkpoint_path = os.path.join(savedir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(self.state_dict(), checkpoint_path)
                print(f'Saved checkpoint at epoch {epoch}.')

        if save_model:
            final_model_path = os.path.join(savedir, f'model_epoch_{num_epochs}.pth')
            torch.save(self.state_dict(), final_model_path)
            print(f'Saved final model at epoch {num_epochs}.')

        return main_metrics

    def _process_outputs(self, inputs, batch_num, instance_num, pooling_method):
        outputs = self(inputs)
        if hasattr(self, 'use_bag_labels') and self.use_bag_labels:
            outputs = outputs.view(batch_num, instance_num, -1)
            if pooling_method == 'mean':
                outputs = torch.mean(outputs, dim=1)
            elif pooling_method == 'max':
                outputs = torch.max(outputs, dim=1)[0]
        return outputs


# ResNet Model
class ResNetModel(AdvancedSetup):
    def __init__(self, num_classes, resnet=18):
        super(ResNetModel, self).__init__()
        if resnet == 152:
            self.resnet = models.resnet152(pretrained=True)
        elif resnet == 50:
            self.resnet = models.resnet50(pretrained=True)
        elif resnet == 18:
            self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x


# DenseNet Model
class DenseNetModel(AdvancedSetup):
    def __init__(self, num_classes):
        super(DenseNetModel, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.densenet(x)
        x = self.fc(x)
        return x


# MIL Model with Custom Pooling
class MILModel(AdvancedSetup):
    def __init__(self, feature_dim, hidden_units, num_classes, pooling='mean'):
        super(MILModel, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_classes)
        self.pooling = pooling

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = x.mean(dim=1) if self.pooling == 'mean' else x.max(dim=1)[0]
        x = self.fc2(x)
        return x


# Vision Transformer (VIT) Model
class VITModel(AdvancedSetup):
    def __init__(self, num_classes, hidden_units=256):
        super(VITModel, self).__init__()
        self.transformer = nn.Transformer(hidden_units, nhead=8, num_encoder_layers=6)
        self.fc = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        x = self.transformer(x, x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


# Capsule Network Model
class CapsuleNet(AdvancedSetup):
    def __init__(self, num_classes, hidden_units=256, num_capsules=10, capsule_dim=8):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=9, stride=1)
        self.primary_capsules = nn.ModuleList([nn.Conv2d(256, capsule_dim, kernel_size=9, stride=2) for _ in range(num_capsules)])
        self.fc = nn.Linear(capsule_dim * num_capsules, hidden_units)
        self.classifier = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        capsules = [capsule(x) for capsule in self.primary_capsules]
        capsules = torch.cat([capsule.view(x.size(0), -1) for capsule in capsules], dim=-1)
        x = torch.relu(self.fc(capsules))
        x = self.classifier(x)
        return x
