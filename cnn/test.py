import torch
import torch.optim as optim
from DataHandler import DataHandler
from CNN import cnn, Trainer
from KNN import KNNTrainer
from perceptron import PerceptronTrainer

def main():
    # 定义参数
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 100
    TRAIN_DATAPATH = r'D:\study\pythonProject1\cnn\Datasets\train'
    TRAIN_TXTPATH = r'D:\study\pythonProject1\cnn\Datasets\train\label.txt'
    TEST_DATAPATH = r'D:\study\pythonProject1\cnn\Datasets\test'
    TEST_TXTPATH = r'D:\study\pythonProject1\cnn\Datasets\test\label.txt'

    # 数据处理
    train_loader, test_loader, inv_label_map = DataHandler.load_data(
        TRAIN_DATAPATH, TRAIN_TXTPATH, TEST_DATAPATH, TEST_TXTPATH, BATCH_SIZE
    )

    # CNN
    # 模型定义
    model = cnn().to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    # 训练和测试
    trainer = Trainer(model, DEVICE, train_loader, test_loader, optimizer, EPOCHS, inv_label_map)
    trainer.run()

    # KNN
    trainer = KNNTrainer(train_loader, test_loader, inv_label_map, n_neighbors=3, pca_components=50)
    trainer.optimize_k()
    trainer.run()

    # 感知机
    trainer = PerceptronTrainer(train_loader, test_loader, inv_label_map, pca_components=50)
    trainer.run(EPOCHS)



main()
