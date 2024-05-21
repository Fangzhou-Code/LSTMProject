import matplotlib.pyplot as plt

# 定义三条直线的起点和终点坐标
line1_start = (0, 0)
line1_end = (5, 5)

line2_start = (5, 5)
line2_end = (10, 0)

line3_start = (10, 0)
line3_end = (0, 0)

# 提取各直线段的起点和终点的 x、y 坐标
line1_start_x, line1_start_y = line1_start
line1_end_x, line1_end_y = line1_end

line2_start_x, line2_start_y = line2_start
line2_end_x, line2_end_y = line2_end

line3_start_x, line3_start_y = line3_start
line3_end_x, line3_end_y = line3_end

# 绘制三条直线段
plt.plot([line1_start_x, line1_end_x], [line1_start_y, line1_end_y], marker='o', linestyle='-', label='Line 1')
plt.plot([line2_start_x, line2_end_x], [line2_start_y, line2_end_y], marker='o', linestyle='-', label='Line 2')
plt.plot([line3_start_x, line3_end_x], [line3_start_y, line3_end_y], marker='o', linestyle='-', label='Line 3')

# 添加起点的标记
plt.plot(line1_start_x, line1_start_y, marker='o', markersize=8, color='green', label='Start Point')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Car Route (Three Straight Lines)')
plt.grid(True)
plt.legend()
plt.show()
