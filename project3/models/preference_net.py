import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    auc,
    precision_recall_curve,
)


class PreferenceNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_criteria):
        super(PreferenceNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.num_criteria = num_criteria

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out
    
    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            X = X.reshape(-1, 1, self.num_criteria)
            out = self.forward(X) 
        out = out.flatten()
        out = list(map(lambda x: [1-x, x], out.tolist()))
        out = np.array(out)
        return out
    
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            X = X.reshape(-1, 1, self.num_criteria)
            out = self.forward(X)
        return out.flatten().item()



def train_model(
    train_loader, test_loader, input_size, hidden_size, num_epochs=50, learning_rate=0.001, path="project3/weights/preference_net.pth", num_criteria=1
):
    
    model = PreferenceNet(input_size, hidden_size, num_criteria=num_criteria)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.01)
    best_accuracy = 0.0  # Track the best accuracy on the test set
    for epoch in (pbar := tqdm(range(num_epochs))):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.float()
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

        # Evaluate on the test set to check accuracy and save the best model
        with torch.no_grad():
            model.eval()
            # use accuracy score as the metric to evaluate the model
            all_predictions = []
            all_targets = []
            for inputs, targets in test_loader:
                outputs = model(inputs)
                predictions = (outputs > 0.5).float().squeeze()
                all_predictions.append(predictions)
                all_targets.append(targets)

            all_predictions = torch.cat(all_predictions)
            all_targets = torch.cat(all_targets)

            accuracy = accuracy_score(all_targets, all_predictions)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), path)

        pbar.set_postfix({"Loss": loss.item(), "Test Accuracy": accuracy})

    return model


def test_model(model, test_loader):
    with torch.no_grad():
        model.eval()
        all_predictions = []
        all_targets = []
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions = (outputs > 0.5).float().squeeze()
            all_predictions.append(predictions)
            all_targets.append(targets)

        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        roc_auc = roc_auc_score(all_targets, all_predictions)

        precision, recall, _ = precision_recall_curve(all_targets, all_predictions)
        pr_auc = auc(recall, precision)

        print(f"Accuracy Score: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Precision-Recall AUC: {pr_auc:.4f}")
