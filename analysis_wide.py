import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



# Read the final data file
df_final = pd.read_excel("output/final_data.xlsx")

# Extracting session types ('CHATBOT' or 'STATIC') from session column
df_final['treatment'] = df_final['session'].apply(lambda x: 'CHATBOT' if 'X_session' in x else 'STATIC')





# BAR CHART: MEAN AVERAGE BELIEF CHANGE BY TREATMENT

# Calculate mean and standard error for each session type
grouped_data = df_final.groupby('treatment')['average_belief_change']
mean_values = grouped_data.mean()
stderr_values = grouped_data.sem()

# Calculate 95% confidence intervals
n = grouped_data.count()
conf_intervals = stderr_values * stats.t.ppf((1 + 0.95) / 2, n - 1)

# Prepare data for plotting
session_types = ['STATIC', 'CHATBOT']  # Reversed order
y_pos = np.arange(len(session_types))
colors = ['blue', 'red']  # Blue for STATIC, Red for CHATBOT

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars with error bars
bars = ax.bar(y_pos, [mean_values[s] for s in session_types], yerr=[conf_intervals[s] for s in session_types],
              align='center', alpha=0.5, capsize=10, color=colors)

# Set labels and title
plt.xticks(y_pos, session_types, fontsize=14)
plt.ylabel('Average belief change', fontsize=14)

# Set y-axis limits
y_max = 50  # Could also be y_max = max(mean_values + conf_intervals) * 1.2
plt.ylim(0, y_max)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)  # Add horizontal grid lines
ax.set_axisbelow(True)  # Ensure grid is drawn behind the bars

# Add mean text on bars
for i, bar in enumerate(bars):
    yval = bar.get_height()
    yerr = conf_intervals[session_types[i]]
    plt.text(bar.get_x() + bar.get_width() / 2, yval + yerr + 0.5,
             f'Mean: {mean_values[session_types[i]]:.2f}', ha='center', va='bottom', fontsize=14)

# Perform t-test
group_chatbot = df_final[df_final['treatment'] == 'CHATBOT']['average_belief_change']
group_static = df_final[df_final['treatment'] == 'STATIC']['average_belief_change']
t_stat, p_value = stats.ttest_ind(group_chatbot, group_static, nan_policy='omit')

# Calculate the difference between the means
mean_difference = mean_values['CHATBOT'] - mean_values['STATIC']

# Add delta symbol with difference value and p-value
delta_y_pos = y_max * 0.95
plt.text(0.5, delta_y_pos, f'Δ: {mean_difference:.2f}, p-value: {p_value:.3f}',
         ha='center', va='center', fontsize=14, color='black')

# Draw connecting lines
horizontal_line_y_pos = delta_y_pos * 0.98
plt.plot([0, 1], [horizontal_line_y_pos, horizontal_line_y_pos], 'k--')

# Calculate the gap (e.g., 5% of the plot height)
gap = y_max * 0.05

for i, xpos in enumerate(y_pos):
    bar_top = mean_values[session_types[i]] + conf_intervals[session_types[i]]
    line_bottom = bar_top + gap
    plt.plot([xpos, xpos], [line_bottom, horizontal_line_y_pos], 'k--')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('output/barchart_average_belief_change.png', dpi=300)
plt.close()




# REGRESSION RESULT: MAIN

import statsmodels.api as sm
import numpy as np

# Create a dummy variable for the session type
df_final['treatment_dummy'] = df_final['treatment'].apply(lambda x: 1 if x == 'CHATBOT' else 0)

# Drop rows with NaN values in the relevant columns
df_regression = df_final[['treatment_dummy', 'average_belief_change']].dropna()

# Prepare the data for regression
X = df_regression['treatment_dummy']
y = df_regression['average_belief_change']

# Add a constant term to the independent variables
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the regression results
print(model.summary())


from statsmodels.iolib.summary2 import summary_col

# Collect results to a single summary table
summary = summary_col(
    [model],
    stars=True,
    model_names=['Main Regression'],
    info_dict={
        'N': lambda x: f"{int(x.nobs):d}",
        'R2': lambda x: f"{x.rsquared:.2f}",
    }
)

# Export to LaTeX
with open('output/main_regression.tex', 'w') as f:
    f.write(summary.as_latex())

# Print the summary to console as well
print(summary)





# BAR CHART: MEAN ABSOLUTE DIVERGENCE BY TREATMENT

# Calculate mean and standard error for each session type
grouped_data = df_final.groupby('treatment')['average_belief_change_absolute']
mean_values = grouped_data.mean()
stderr_values = grouped_data.sem()

# Calculate 95% confidence intervals
n = grouped_data.count()
conf_intervals = stderr_values * stats.t.ppf((1 + 0.95) / 2, n - 1)

# Prepare data for plotting
session_types = ['STATIC', 'CHATBOT']  # Reversed order
y_pos = np.arange(len(session_types))
colors = ['blue', 'red']  # Blue for STATIC, Red for CHATBOT

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars with error bars
bars = ax.bar(y_pos, [mean_values[s] for s in session_types], yerr=[conf_intervals[s] for s in session_types],
              align='center', alpha=0.5, capsize=10, color=colors)

# Set labels and title
plt.xticks(y_pos, session_types, fontsize=14)
plt.ylabel('Mean Absolute Divergence', fontsize=14)

# Set y-axis limits
y_max = max(mean_values + conf_intervals) * 1.2  # Could also be y_max = max(mean_values + conf_intervals) * 1.2
plt.ylim(0, y_max)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)  # Add horizontal grid lines
ax.set_axisbelow(True)  # Ensure grid is drawn behind the bars

# Add mean text on bars
for i, bar in enumerate(bars):
    yval = bar.get_height()
    yerr = conf_intervals[session_types[i]]
    plt.text(bar.get_x() + bar.get_width() / 2, yval + yerr + 0.5,
             f'Mean: {mean_values[session_types[i]]:.2f}', ha='center', va='bottom', fontsize=14)

# Perform t-test
group_chatbot = df_final[df_final['treatment'] == 'CHATBOT']['average_belief_change_absolute']
group_static = df_final[df_final['treatment'] == 'STATIC']['average_belief_change_absolute']
t_stat, p_value = stats.ttest_ind(group_chatbot, group_static, nan_policy='omit')

# Calculate the difference between the means
mean_difference = mean_values['CHATBOT'] - mean_values['STATIC']

# Add delta symbol with difference value and p-value
delta_y_pos = y_max * 0.95
plt.text(0.5, delta_y_pos, f'Δ: {mean_difference:.2f}, p-value: {p_value:.3f}',
         ha='center', va='center', fontsize=14, color='black')

# Draw connecting lines
horizontal_line_y_pos = delta_y_pos * 0.98
plt.plot([0, 1], [horizontal_line_y_pos, horizontal_line_y_pos], 'k--')

# Calculate the gap (e.g., 5% of the plot height)
gap = y_max * 0.05

for i, xpos in enumerate(y_pos):
    bar_top = mean_values[session_types[i]] + conf_intervals[session_types[i]]
    line_bottom = bar_top + gap
    plt.plot([xpos, xpos], [line_bottom, horizontal_line_y_pos], 'k--')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('output/barchart_absolute_divergence.png', dpi=300)
plt.close()


