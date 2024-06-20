import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)  # 1: 输入通道，10: 输出通道，5: kernel大小
        self.conv2 = nn.Conv2d(10, 20, 3)  # 10: 输入通道，20: 输出通道，3: kernel大小
        self.fc1 = nn.Linear(20 * 62 * 62, 500)
        self.fc2 = nn.Linear(500, 10)  # 500: 输入通道，10: 输出通道

    def forward(self, x):
        input_size = x.size(0)  # batch_size
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)    # ---10,126,126
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)    # ---20,62,62
        x = x.view(input_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Trainer:
    def __init__(self, model, device, train_loader, test_loader, optimizer, epochs, inv_label_map):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.inv_label_map = inv_label_map

    def train_model(self, epoch):
        self.model.train()
        for batch_index, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_index % 3000 == 0:
                print(f"Train Epoch : {epoch} \t Loss : {loss.item():.6f}")

    def test_model(self):
        self.model.eval()
        correct = 0.0
        test_loss = 0.0
        all_preds = []
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                all_preds.extend(pred.cpu().numpy())

            test_loss /= len(self.test_loader.dataset)
            print(
                f"Test -- Average Loss : {test_loss:.4f}, Accuracy : {correct / len(self.test_loader.dataset) * 100.0:.3f}")

        # Print predictions
        for i, pred in enumerate(all_preds):
            print(f"Image {i + 1}: Predicted - {self.inv_label_map[pred[0]]}")

    def run(self):
        for epoch in range(1, self.epochs + 1):
            self.train_model(epoch)
            self.test_model()
