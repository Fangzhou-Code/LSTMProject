import matplotlib.pyplot as plt

# Data for training and fine-tuning times for different numbers of forklifts
num_forklift = [10, 20, 30, 40, 50]
train_time = [66.83700180053711, 127.06777715682983, 187.54643630981445, 243.24084663391113, 301.9783582687378]
fine_tune_time = [23.15790557861328, 46.37213468551636, 70.00183939933777, 92.75068521499634, 115.406121969223]

bar_width = 0.35
index = range(len(num_forklift))

# Create a figure and set of axes
fig, ax = plt.subplots()

# Nature style colors (simple, toned down, soft palette)
colors_nature = ['#2E91E5', '#E15F99']

# Plotting the bars for training and fine-tuning times with Nature-inspired colors
ax.bar(index, train_time, bar_width, label='Training Time', color=colors_nature[0])
ax.bar([i + bar_width for i in index], fine_tune_time, bar_width, label='Fine-Tuning Time', color=colors_nature[1])

# Labels and formatting
ax.set_xlabel('Number of Forklifts')
ax.set_ylabel('Time (seconds)')
ax.set_title('Training and Fine-Tuning Time Comparison')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(num_forklift)

# Displaying legend
ax.legend()

# Minimizing empty space
plt.tight_layout()

# Show the plot
plt.show()
