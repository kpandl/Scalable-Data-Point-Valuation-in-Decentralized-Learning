import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path_cost_institution = os.path.join(os.getcwd(), "measurement_results", "public_transaction_cost_institution.csv")
file_path_cost_orchestrator = os.path.join(os.getcwd(), "measurement_results", "public_transaction_cost_orchestrator.csv")

df_cost_institution = pd.read_csv(file_path_cost_institution)
df_cost_orchestrator = pd.read_csv(file_path_cost_orchestrator)

column_name = "gas_used"

bar_starts_orchestrator = []
bar_ends_orchestrator = []

# iterate over df_cost_orchestrator
running_end_orchestrator = 0
for index, row in df_cost_orchestrator.iterrows():
    bar_starts_orchestrator.append(running_end_orchestrator)
    running_end_orchestrator += row[column_name]
    bar_ends_orchestrator.append(running_end_orchestrator)

bar_starts_institution = []
bar_ends_institution = []

# iterate over df_cost_institution
running_end_institution = 0
for index, row in df_cost_institution.iterrows():
    bar_starts_institution.append(running_end_institution)
    running_end_institution += row[column_name]
    bar_ends_institution.append(running_end_institution)

# Plot the stacked bar chart
fig, ax = plt.subplots()

bar_width = 0.35
opacity = 0.8

index = np.arange(len(df_cost_institution))

# Define colors for the bars
institution_colors = ["#5a40cf"]
orchestrator_colors = ["#032152", "#709be0", "#70cae0"]

orchestrator_labels = ["contract creation", "contract funding", "contract payout"]

institution_bars = ax.bar(index, bar_ends_institution, bar_width, bottom=bar_starts_institution, label="Institution, valuation\nresults posting", alpha=opacity, color=institution_colors)
for i in range(3):
    print("starting at: " + str(bar_starts_orchestrator[i]), "ending at: " + str(bar_ends_orchestrator[i]))
    orchestrator_bars = ax.bar(index + bar_width + 0.1, bar_ends_orchestrator[i]-bar_starts_orchestrator[i], bar_width, bottom=bar_starts_orchestrator[i], label="Orchestrator,\n"+orchestrator_labels[i], alpha=opacity, color=orchestrator_colors[i])

# Add x-ticks
ax.set_xticks([index[0] + 0 / 2, index[-1] + bar_width / 2 + bar_width / 2 + 0.1])
ax.set_xticklabels(["Institution", "Orchestrator"])

# Add labels, title, and legend
ax.set_xlabel("Role")
ax.set_ylabel("Gas cost")
ax.set_title("Gas cost of public transactions per type")

# Add legend
plt.legend()

# set size of plot
width_inch_from_cm = 13 / 2.54
height_inch_from_cm = 9 / 2.54
fig.set_size_inches(width_inch_from_cm, height_inch_from_cm)

plt.tight_layout()

# Show the figure
plt.show()
