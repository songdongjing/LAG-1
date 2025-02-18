import os

# 文件夹路径
folder_path = "scripts\\results\\SingleCombat\\1v1\\ShootMissile\\Selfplay\\dsac\\v1"

# 获取文件夹中的所有文件名
filenames = os.listdir(folder_path)
# # 筛选出所有 TensorBoard 文件
# tensorboard_files = [f for f in filenames if f.endswith(".tfevents")]

# logdir = ",".join([os.path.join(folder_path, f) for f in tensorboard_files])
# print(logdir)
os.system(f"tensorboard --logdir={folder_path}")