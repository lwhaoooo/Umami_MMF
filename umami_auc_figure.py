import pdb

import matplotlib.pyplot as plt
import numpy as np

# 加载不同模型的 ROC 数据
roc_data_gin = np.load("/root/autodl-tmp/new_protein_graph/data/umami/auc_figure/gin1_roc_data.npy", allow_pickle=True).item()
roc_data_gat_gcn = np.load("/root/autodl-tmp/new_protein_graph/data/umami/auc_figure/gat_gcn_roc_data.npy", allow_pickle=True).item()
roc_data_gat = np.load("/root/autodl-tmp/new_protein_graph/data/umami/auc_figure/gat_roc_data.npy", allow_pickle=True).item()
roc_data_gcn = np.load("/root/autodl-tmp/new_protein_graph/data/umami/auc_figure/gcn_roc_data.npy", allow_pickle=True).item()

# roc_data_gin = np.load("/root/autodl-tmp/new_protein_graph/data/umami/auc_figure/ump789_gin_roc_data.npy", allow_pickle=True).item()
# roc_data_gat_gcn = np.load("/root/autodl-tmp/new_protein_graph/data/umami/auc_figure/ump789_gat_gcn_roc_data.npy", allow_pickle=True).item()
# roc_data_gat = np.load("/root/autodl-tmp/new_protein_graph/data/umami/auc_figure/ump789_gat_roc_data.npy", allow_pickle=True).item()
# roc_data_gcn = np.load("/root/autodl-tmp/new_protein_graph/data/umami/auc_figure/ump789_gcn_roc_data.npy", allow_pickle=True).item()

# pdb.set_trace()

# 访问 FPR, TPR 和 AUC
best_fpr_model_gin = roc_data_gin["fpr"]
best_tpr_model_gin = roc_data_gin["tpr"]
best_auc_model_gin = roc_data_gin["best_auc"]

best_fpr_model_gat_gcn = roc_data_gat_gcn["fpr"]
best_tpr_model_gat_gcn = roc_data_gat_gcn["tpr"]
best_auc_model_gat_gcn = roc_data_gat_gcn["best_auc"]

best_fpr_model_gat = roc_data_gat["fpr"]
best_tpr_model_gat = roc_data_gat["tpr"]
best_auc_model_gat = roc_data_gat["best_auc"]

best_fpr_model_gcn = roc_data_gcn["fpr"]
best_tpr_model_gcn = roc_data_gcn["tpr"]
best_auc_model_gcn = roc_data_gcn["best_auc"]

# pdb.set_trace()

# 绘制 ROC 曲线
plt.figure(figsize=(10, 8))  # 设置图像大小

# 绘制 Model 1 KAN 的 ROC 曲线
plt.plot(best_fpr_model_gin, best_tpr_model_gin, label=f"GIN (AUC={best_auc_model_gin:.4f})",
         color="blue", linewidth=1.5, linestyle='-', alpha=0.8)

plt.plot(best_fpr_model_gat_gcn, best_tpr_model_gat_gcn, label=f"GAT_GCN (AUC={best_auc_model_gat_gcn:.4f})",
         color="green", linewidth=1.5, linestyle='-', alpha=0.8)

plt.plot(best_fpr_model_gat, best_tpr_model_gat, label=f"GAT (AUC={best_auc_model_gat:.4f})",
         color="darkorange", linewidth=1.5, linestyle='-', alpha=0.8)

plt.plot(best_fpr_model_gcn, best_tpr_model_gcn, label=f"GCN (AUC={best_auc_model_gcn:.4f})",
         color="red", linewidth=1.5, linestyle='-', alpha=0.8)

# 绘制随机猜测的参考线
plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC=0.5000)", linewidth=1)

# 设置标签和标题
plt.xlabel("False Positive Rate", fontsize=18)
plt.ylabel("True Positive Rate", fontsize=18)
plt.title("ROC Curve Comparison of Four GNN Models with AUC Scores", fontsize=18)

# 增大坐标轴数值字体
plt.tick_params(axis='both', which='major', labelsize=18)  # 增大坐标轴数值字体

# 显示图例，增加图例的字体大小
plt.legend(loc="lower right", fontsize=16)

# 设置网格，增强可读性
plt.grid(True, linestyle="--", alpha=0.7)

# # 保存图像到指定路径
# output_path = "/root/autodl-tmp/new_protein_graph/data/umami/auc_figure/ump789_graph_roc_curve.png"
# plt.savefig(output_path, format="png", dpi=300)

# 保存图像为 PDF 格式，并去除白边
output_path = "/root/autodl-tmp/new_protein_graph/data/umami/auc_figure/ump442_graph_roc_curve.pdf"
plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)

# 显示图像
plt.show()

print(f"ROC Curve saved at {output_path}")