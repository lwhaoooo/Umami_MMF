import numpy as np
import pandas as pd
import sys, os  # 用于系统和文件操作
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef
import pdb


# # 在每个 epoch 训练模型的函数
# def train(model, device, train_loader, optimizer, epoch):
#     print('Training on {} samples...'.format(len(train_loader.dataset)))
#     model.train()  # 将模型设置为训练模式
#     for batch_idx, data in enumerate(train_loader):
#         data = data.to(device)  # 将数据移到指定的设备上
#         optimizer.zero_grad()  # 清除梯度
#         output = model(data).view(-1)  # 修改：调整输出维度以匹配 BCEWithLogitsLoss
#         loss = loss_fn(output, data.y.float().to(device))  # 修改：使用 BCEWithLogitsLoss 需要将标签转为 float 类型
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新模型参数
#         if batch_idx % LOG_INTERVAL == 0:
#             print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
#                                                                            batch_idx * len(data.x),
#                                                                            len(train_loader.dataset),
#                                                                            100. * batch_idx / len(train_loader),
#                                                                            loss.item()))
# def predicting(model, device, loader):
#     model.eval()  # 将模型设置为评估模式
#     total_preds = torch.Tensor()  # 存储所有预测结果
#     total_labels = torch.Tensor()  # 存储所有真实标签
#     print('Make prediction for {} samples...'.format(len(loader.dataset)))
#     with torch.no_grad():  # 关闭梯度计算
#         for data in loader:
#             data = data.to(device)  # 将数据移到指定的设备上
#             output = model(data).view(-1)  # 修改：调整输出维度以匹配 BCEWithLogitsLoss
#             preds = torch.sigmoid(output)  # 修改：使用 Sigmoid 将输出转为概率值
#             # preds = (preds > 0.5).float()  # 修改：将概率值转为二分类结果
#             total_preds = torch.cat((total_preds, preds.cpu()), 0)  # 记录预测结果
#             total_labels = torch.cat((total_labels, data.y.cpu()), 0)  # 记录真实标签
#     return total_labels.numpy().flatten(), total_preds.numpy().flatten()

# t-SNE 可视化函数，修改为淡色节点、加上图例、调整标签字体大小
def plot_tsne(features, labels, epoch, save_dir="/root/autodl-tmp/new_protein_graph/data/umami/t_sne_1"):
    # features 已经是 numpy 数组，所以不需要 detach 和 cpu
    X = features  # 如果 features 是 numpy 数组，直接使用即可
    y = labels.astype(int)  # 确保标签为整数

    # 初始化 t-SNE
    tsne = TSNE(n_components=2, init='random', random_state=1, learning_rate=200.0)
    X_tsne = tsne.fit_transform(X)

    # 归一化处理
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = X_tsne

    # 保存降维后的特征和标签
    np.savetxt(os.path.join(save_dir, "Embeds_{}_epoch_{}.txt".format("DPI", epoch)), X_norm)
    np.savetxt(os.path.join(save_dir, "labels_{}_epoch_{}.txt".format("DPI", epoch)), y)

    # 绘制散点图
    plt.figure(figsize=(15, 15))
    plt.xlim([x_min[0] - 5, x_max[0] + 5])
    plt.ylim([x_min[1] - 5, x_max[1] + 5])

    # 定义淡黄色和淡蓝色
    colors = ['#ffdd89', '#87ceeb']  # 使用更淡的颜色，黄色和浅蓝
    labels_names = ['Non-Umami', 'Umami']

    # 绘制每个类别的散点图，并为其添加标签
    for i in range(len(X_norm)):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=colors[y[i]], alpha=0.7)

    # 添加图例，并调整图例字体大小
    scatter1 = plt.scatter([], [], color=colors[0], label=labels_names[0], alpha=0.7)
    scatter2 = plt.scatter([], [], color=colors[1], label=labels_names[1], alpha=0.7)
    plt.legend(handles=[scatter1, scatter2], loc='upper left', fontsize=18)

    # # 设置标题、横轴和纵轴标签，并调整字体大小
    # plt.title(f'Model 1 Scatter Plot at Epoch {epoch}', fontsize=20)

    # 调整坐标轴刻度数字的字体大小
    plt.tick_params(axis='both', which='major', labelsize=18)

    # 保存图像为 .png 文件，并去除白边
    # plt.savefig(os.path.join(save_dir, f"model_1_tsne_DPI_epoch_{epoch}.png"), bbox_inches='tight')
    # print(f"t-SNE plot saved at: {os.path.join(save_dir, f'model_1_tsne_DPI_epoch_{epoch}.png')}")
    # plt.close()  # 关闭当前图表以避免内存问题

    # 保存图像为 .pdf 文件，并去除白边
    pdf_filename = os.path.join(save_dir, f"ump789_tsne_DPI_epoch_{epoch}.pdf")
    plt.savefig(pdf_filename, bbox_inches='tight', format='pdf')  # 保存为 PDF 格式
    print(f"t-SNE plot saved at: {pdf_filename}")
    plt.close()  # 关闭当前图表以避免内存问题


def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    all_features = []
    all_labels = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, features = model(data)  # 提取中间特征
        loss = loss_fn(output.view(-1), data.y.float().to(device))
        loss.backward()
        optimizer.step()

        # 收集中间层特征用于 t-SNE 可视化
        all_features.append(features.cpu().detach().numpy())
        all_labels.append(data.y.cpu().detach().numpy())

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data.x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    # t-SNE 可视化，仅在 epoch 1、10、100 时保存图片
    if epoch in [1, 10, 50]:
        all_features = np.vstack(all_features)
        all_labels = np.hstack(all_labels)
        print(f"Features shape: {all_features.shape}, Labels shape: {all_labels.shape}")
        plot_tsne(all_features, all_labels, epoch)


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # 解包模型的输出（假设返回的是 (output, features)）
            output, _ = model(data)  # 只使用 output 进行预测
            output = output.view(-1)  # 保证输出的维度正确
            preds = torch.sigmoid(output)
            # preds = (preds > 0.5).float()
            total_preds = torch.cat((total_preds, preds.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def evaluate_performance(y_true, y_pred):
    # pdb.set_trace()
    y_pred_labels = (y_pred >= 0.5).astype(int)  # 修改：将概率转换为二值标签
    auc = roc_auc_score(y_true, y_pred)
    prc = average_precision_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    accuracy = accuracy_score(y_true, y_pred_labels)

    return auc, prc, precision, recall, accuracy


datasets = [['ump442'], ['ump789']][int(sys.argv[1])]  # 选择数据集
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]  # 选择模型
model_st = modeling.__name__

cuda_name = "cuda:0"  # 默认使用 cuda:0 作为 GPU 设备
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 32  # 训练批次大小
TEST_BATCH_SIZE = 32  # 测试批次大小
LR = 0.001  # 学习率
LOG_INTERVAL = 20  # 日志记录间隔
NUM_EPOCHS = 100  # 训练的 epoch 数

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# 主程序：迭代处理不同的数据集
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)
    # pdb.set_trace()
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset + '_train')  # 加载预处理后的数据
        test_data = TestbedDataset(root='data', dataset=dataset + '_test')

        # 将数据划分为训练集和验证集 (80% 训练集，20% 验证集)
        train_size = int(0.8 * len(train_data))  # 80% 作为训练集
        valid_size = len(train_data) - train_size  # 剩余 20% 作为验证集
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

        # 将数据准备为 PyTorch mini-batch 处理格式
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # 训练模型
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)  # 初始化模型并移到指定设备上
        loss_fn = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss 作为损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 使用 Adam 优化器
        best_auc = 0  # 初始化最佳AUC
        best_epoch = -1  # 初始化最佳epoch
        model_file_name = 'umami_' + 'model_' + model_st + '_' + dataset + '_eval' + '.model'  # 模型文件名
        result_file_name = 'umami_' + 'result_' + model_st + '_' + dataset + '_eval' + '.csv'  # 结果文件名

        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch + 1)  # 训练模型
            print('Validating for validation data')
            G, P = predicting(model, device, valid_loader)  # 在验证集上进行预测
            auc_score, prc_score, precision, recall, accuracy = evaluate_performance(G, P)  # 计算评估指标
            print(
                f'Epoch {epoch + 1}: AUC: {auc_score:.4f}, PRC: {prc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

            # 保存最佳模型
            if auc_score > best_auc:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, [auc_score, prc_score, precision, recall, accuracy])))
                best_auc = auc_score
                best_epoch = epoch + 1
                print('AUC improved at epoch ', best_epoch, '; best_auc:', best_auc, model_st, dataset)
            else:
                print('No improvement since epoch ', best_epoch, '; best_auc:', best_auc, model_st, dataset)
