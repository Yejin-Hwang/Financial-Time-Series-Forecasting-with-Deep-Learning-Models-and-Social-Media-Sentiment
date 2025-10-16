import matplotlib.pyplot as plt
import numpy as np

# Data
stocks = ['TSLA', 'NVDA']
rmse_reduction = [66.9, 69.2]
mape_reduction = [71.8, 72.7]

x = np.arange(len(stocks))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))

# Bars
bars1 = ax.bar(x - width/2, rmse_reduction, width, label='RMSE ↓ (%)', color='#1f77b4')
bars2 = ax.bar(x + width/2, mape_reduction, width, label='MAPE ↓ (%)', color='#ff7f0e')

# Labels
ax.set_ylabel('Error Reduction (%)')
ax.set_title('Performance Improvement: TFT_Reddit vs TFT_Baseline')
ax.set_xticks(x)
ax.set_xticklabels(stocks)
ax.legend()

# Adjust y-axis range (add headroom)
ax.set_ylim(0, max(max(rmse_reduction), max(mape_reduction)) + 10)

# Annotate each bar (shifted a bit higher)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 2.5,   # <-- moved slightly higher
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=9
        )

plt.tight_layout()
plt.savefig('results/tft_improvement.png', dpi=300)
plt.show()
