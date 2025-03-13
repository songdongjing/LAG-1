import matplotlib.pyplot as plt
import numpy as np
import ast

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def load_winrate_data(filename):
    episodes = []
    winrates = []
    with open(filename, 'r') as f:
        for line in f:
            # 使用ast.literal_eval安全地解析字符串列表
            data = ast.literal_eval(line.strip())
            episodes.append(data[0])
            winrates.append(data[1])
    return np.array(episodes), np.array(winrates)

def plot_winrate(episodes, winrates):
    plt.figure(figsize=(12, 6))
    
    # 绘制胜率曲线
    plt.plot(episodes, winrates, 'b-', label='胜率', linewidth=2)
    
    # 设置图表属性
    plt.title('智能体对战胜率曲线', fontsize=14)
    plt.xlabel('回合数', fontsize=12)
    plt.ylabel('胜率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 设置y轴范围为0-1
    plt.ylim(-0.1, 1.1)
    
    # 保存图片
    plt.savefig('winrate_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载数据
    episodes, winrates = load_winrate_data('winrate_list.txt')
    
    # 绘制图表
    plot_winrate(episodes, winrates)
    print("胜率曲线图已保存为 'winrate_curve.png'")

if __name__ == "__main__":
    main()
