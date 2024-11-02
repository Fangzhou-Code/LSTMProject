import matplotlib.pyplot as plt
import numpy as np

# Data
per_control = [0.3, 0.5, 0.6, 0.8]
device_fingerprint_acc = [0.7574000000000001, 0.6036666666666667, 0.5195333333333334 ,0.3626]
lstm_acc = [0.6243000775575638, 0.561800017952919, 0.555000051856041,   0.539500042796135]
lstm_transfer_acc = [0.9165000879764557, 0.9274001181125641, 0.9302000045776367, 0.9477000463008881]
retrain_acc = [0.9643001019954681, 0.9706000745296478, 0.9827000045776367, 0.9835000383853912]

bar_width = 0.2
index = np.arange(len(per_control))

# Create a figure and set of axes
fig, ax = plt.subplots()
plt.ylim(0, 1.1)
ax.set_yticks([0.1 * i for i in range(11)])


# Nature style colors (simple, toned down, soft palette)
colors = ['#8EA0C9',  '#E889BD', '#FC8C63', '#67C2A3']

# Plot each set of bars
# Plot each set of bars with the correct order and swapped colors for 'Retrain' and 'Device Fingerprint'
ax.bar(index, device_fingerprint_acc, bar_width, label='Device Fingerprint', color=colors[0])
ax.bar(index + bar_width, lstm_acc, bar_width, label='LSTM', color=colors[2])
ax.bar(index + 2 * bar_width, lstm_transfer_acc, bar_width, label='Transfer Learning Based', color=colors[3])
ax.bar(index + 3 * bar_width, retrain_acc, bar_width, label='Retraining', color=colors[1])

# Labels and formatting

ax.set_ylabel('Accuracy')
ax.set_title('Average Accuracy Comparison')  # Updated title

ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(per_control)

# Adjusting legend position slightly lower but ensuring it doesn't overlap the bars
ax.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2, frameon=False)

# Minimizing empty space
plt.tight_layout()

# Show the plot
plt.show()
