import torch
import torch.optim as optim
from cnn.DataHandler.DataHandler import DataHandler
from cnn.NeuralNetwork.CNN import cnn, CNN_Trainer


def main():
    # 定义参数
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 100
    LEARNING_RATE = 0.01
    TRAIN_DATAPATH = r'D:\study\pythonProject1\cnn\Datasets\train'
    TRAIN_TXTPATH = r'D:\study\pythonProject1\cnn\Datasets\train\label.txt'
    TEST_DATAPATH = r'D:\study\pythonProject1\cnn\Datasets\test'
    TEST_TXTPATH = r'D:\study\pythonProject1\cnn\Datasets\test\label.txt'

    # 数据处理
    train_loader, test_loader, inv_label_map = DataHandler.form_data(
        TRAIN_DATAPATH, TRAIN_TXTPATH, TEST_DATAPATH, TEST_TXTPATH, BATCH_SIZE
    )
    # CNN
    # 模型定义
    model = cnn().to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    # 训练和测试
    trainer = CNN_Trainer(model, DEVICE, train_loader, test_loader, optimizer, EPOCHS, inv_label_map)
    trainer.run()

    # KNN
    # trainer = KNNTrainer(train_loader, test_loader, inv_label_map, n_neighbors=3, pca_components=50)
    # trainer.optimize_k()
    # trainer.run()
    #
    # # 感知机
    # trainer = PerceptronTrainer(train_loader, test_loader, inv_label_map, pca_components=50)
    # trainer.run(EPOCHS)
    #
    # # BP
    # model = SimpleNN().to(DEVICE)
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # trainer = BP_Trainer(model, DEVICE, train_loader, test_loader, optimizer, EPOCHS, inv_label_map)
    # trainer.run()




main()
