import matplotlib.pyplot as plt
import numpy as np

# Data
per_control = [0.2, 0.4, 0.6, 0.8, 1.0]
device_fingerprint_acc = [0.8422, 0.6816666666666666, 0.5195333333333334, 0.3626, 0.2]
lstm_acc = [0.6392000466585159, 0.5790000408887863, 0.561800017952919, 0.539500042796135, 0.550900012254715]
lstm_transfer_acc = [0.8187000155448914, 0.8395000696182251, 0.7802000045776367, 0.7677000463008881, 0.7761000096797943]
retrain_acc = [0.868800014257431, 0.8637000620365143, 0.8427000045776367, 0.8535000383853912, 0.8460000455379486]

bar_width = 0.2
index = np.arange(len(per_control))

# Create a figure and set of axes
fig, ax = plt.subplots()
plt.ylim(0,1)

# Nature style colors (simple, toned down, soft palette)
colors = ['#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D']

# Plot each set of bars
# Plot each set of bars with the correct order and swapped colors for 'Retrain' and 'Device Fingerprint'
ax.bar(index, device_fingerprint_acc, bar_width, label='Device Fingerprint', color=colors[3])
ax.bar(index + bar_width, lstm_acc, bar_width, label='LSTM', color=colors[1])
ax.bar(index + 2 * bar_width, lstm_transfer_acc, bar_width, label='LSTM + Transfer Learning', color=colors[2])
ax.bar(index + 3 * bar_width, retrain_acc, bar_width, label='Retrain', color=colors[0])

# Labels and formatting
ax.set_xlabel('Percentage Control')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Comparison')  # Updated title

ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(per_control)

# Adjusting legend position slightly lower but ensuring it doesn't overlap the bars
ax.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2, frameon=False)

# Minimizing empty space
plt.tight_layout()

# Show the plot
plt.show()
