import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

torch.manual_seed(1)
torch.cuda.manual_seed(1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None  # will be created during fit
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        self.device = "cpu"
        
    def fit(self, X, y, epochs=20):
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        in_size = X_tensor.shape[1]
        out_size = int(torch.max(y_tensor).item()) + 1

        self.model = nn.Sequential(
                    nn.Linear(in_size, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256,64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64,out_size)
                ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()
            

        dataset = TensorDataset(X_tensor, y_tensor)
        if y_tensor.shape[0] >= 10000:
            dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
        else:
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        self.model.train()
        for epoch in range(epochs):
            for x, y in tqdm(dataloader, unit="batch"):
                x, y = x.float().to(self.device), y.long().to(self.device)
                self.optimizer.zero_grad() # Remove the gradients from the previous step
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

    def forward(self, X):
        return self.model(X)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(next(self.model.parameters()).device)
            pred = self.model(X_tensor)
            y_pred = torch.argmax(pred, dim=1)

        return  y_pred.cpu().numpy()
    
    def model_size(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)