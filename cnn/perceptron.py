import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

class PerceptronTrainer:
    def __init__(self, train_loader, test_loader, inv_label_map, pca_components=50):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.inv_label_map = inv_label_map
        self.model = Perceptron(max_iter=1000, tol=1e-3)
        self.pca_components = pca_components
        self.pca = None

    def prepare_data(self, loader):
        data = []
        targets = []
        for images, labels in loader:
            images = images.view(images.size(0), -1).numpy()
            data.extend(images)
            targets.extend(labels.numpy())
        return np.array(data), np.array(targets)

    def fit_pca(self, data):
        n_samples, n_features = data.shape
        n_components = min(self.pca_components, n_samples, n_features)
        self.pca = PCA(n_components=n_components)
        return self.pca.fit_transform(data)

    def train(self):
        train_data, train_targets = self.prepare_data(self.train_loader)
        train_data = self.fit_pca(train_data)
        self.model.fit(train_data, train_targets)
        print("Training complete.")

    def test(self):
        test_data, test_targets = self.prepare_data(self.test_loader)
        test_data = self.pca.transform(test_data)
        predictions = self.model.predict(test_data)
        accuracy = accuracy_score(test_targets, predictions)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        for i, pred in enumerate(predictions):
            print(f"Image {i + 1}: Predicted - {self.inv_label_map[pred]}")

    def run(self, epochs):
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}")
            self.train()
            self.test()