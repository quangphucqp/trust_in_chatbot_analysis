import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df_final = pd.read_excel("output/final_data_unshuffled.xlsx")
df_final['treatment'] = df_final['session'].apply(lambda x: 'CHATBOT' if 'X_session' in x else 'STATIC')

# Dictionary of correct answers:
correct_answer_dict = {
    1: "B",
    2: "C",
    3: "D",
    4: "A",
    5: "C"
}

### GENERATE LONG FORMAT DATAFRAME ###

# Step 1: Select the required columns
cols_to_keep = ['computer_number', 'session', 'treatment']
questions_cols = [col for col in df_final.columns if col.startswith('questions_seq_groupmove')]
selected_df = df_final[cols_to_keep + questions_cols]

# Step 2: Reshape the dataframe
# Melt the dataframe to create a long format
melted_df = pd.melt(selected_df,
                    id_vars=['computer_number', 'session', 'treatment'],
                    var_name='original_column',
                    value_name='value')

# Step 3: Extract question number and xxx part from the original column name
melted_df[['prefix', 'question_number', 'player', 'xxx']] = melted_df['original_column'].str.split('.', n=3, expand=True)

# Step 4: Create one-hot encoded columns for questions
for i in range(1, 6):
    melted_df[f'Q{i}'] = (melted_df['question_number'] == str(i)).astype(int)

# Step 5: Pivot the dataframe to create separate columns for each 'xxx' part
pivoted_df = melted_df.pivot_table(
    index=['computer_number', 'session', 'treatment', 'question_number'] + [f'Q{i}' for i in range(1, 6)],
    columns='xxx',
    values='value',
    aggfunc='first'
).reset_index()

# Flatten the column names
pivoted_df.columns.name = None

# Step 6: Reorder columns for better readability
static_cols = ['computer_number', 'session', 'treatment', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
dynamic_cols = [col for col in pivoted_df.columns if col not in static_cols]
column_order = static_cols + dynamic_cols
final_df = pivoted_df[column_order]

# Reset index for the final dataframe
final_df = final_df.reset_index(drop=True)


# Combine with rec data:

def wide_to_long(df, id_vars):
    # Identify the columns to be melted
    value_vars = [col for col in df.columns if col not in id_vars]

    # Melt the dataframe
    df_long = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='original_column', value_name='value')

    # Extract the question number and create one-hot encoded columns
    df_long['question_number'] = df_long['original_column'].str.extract(r'\.(\d)$')
    for i in range(1, 6):
        df_long[f'Q{i}'] = (df_long['question_number'] == str(i)).astype(int)

    # Extract the measure name (everything before the last dot)
    df_long['measure'] = df_long['original_column'].str.rsplit('.', n=1).str[0]

    # Pivot the dataframe to create separate columns for each measure
    df_wide = df_long.pivot_table(
        index=id_vars + ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
        columns='measure',
        values='value',
        aggfunc='first'
    ).reset_index()

    # Flatten the column names
    df_wide.columns.name = None

    return df_wide


# Read and convert O_rec_unshuffled to long format
O_rec_unshuffled = pd.read_excel("output/O_rec_unshuffled.xlsx")
O_rec_long = wide_to_long(O_rec_unshuffled, id_vars=['session', 'computer_number'])

# Read and convert X_chatlog_unshuffled to long format
X_chatlog_unshuffled = pd.read_excel("output/X_chatlog_unshuffled.xlsx")
X_chatlog_long = wide_to_long(X_chatlog_unshuffled, id_vars=['session', 'computer_number'])

# Concatenate O_rec_long and X_chatlog_long
combined_data_long = pd.concat([O_rec_long, X_chatlog_long], ignore_index=True)

# Merge final_df with combined_data_long
df_final_long = pd.merge(final_df, combined_data_long,
                         how='inner',
                         on=['session', 'computer_number', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

# If there are any duplicate columns from the merge, we can drop them
df_final_long = df_final_long.loc[:, ~df_final_long.columns.duplicated()]

# Reorder columns for better readability
id_cols = ['session', 'computer_number', 'treatment', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
other_cols = [col for col in df_final_long.columns if col not in id_cols]
df_final_long = df_final_long[id_cols + other_cols]

# Reset index
df_final_long = df_final_long.reset_index(drop=True)


# Calculate pre and post belief:
def get_belief(row, prefix):
    if row['rec_label'] in ['A', 'B', 'C', 'D']:
        return row[f'{prefix}_belief_{row["rec_label"].lower()}']
    else:
        return np.nan

# Create the pre_belief column
df_final_long['pre_belief'] = np.nan
for i in range(1, 6):
    mask = df_final_long[f'Q{i}'] == 1
    df_final_long.loc[mask, 'pre_belief'] = df_final_long.loc[mask].apply(get_belief, axis=1, prefix='pre')

# Create the post_belief column
df_final_long['post_belief'] = np.nan
for i in range(1, 6):
    mask = df_final_long[f'Q{i}'] == 1
    df_final_long.loc[mask, 'post_belief'] = df_final_long.loc[mask].apply(get_belief, axis=1, prefix='post')

# Indicator of recommendation being correct:
df_final_long['question_number'] = df_final_long['question_number'].astype(int)
df_final_long['correct_answer'] = df_final_long['question_number'].map(correct_answer_dict)
df_final_long['rec_correct'] = df_final_long['rec_label'] == df_final_long['correct_answer']

# Create new columns for pre_belief and post_belief based on the correct answer
df_final_long['pre_belief_correct'] = df_final_long.apply(
    lambda row: row[f'pre_belief_{row["correct_answer"].lower()}'], axis=1
)
df_final_long['post_belief_correct'] = df_final_long.apply(
    lambda row: row[f'post_belief_{row["correct_answer"].lower()}'], axis=1
)

# Add participant id
df_final_long['participant_id'] = df_final_long['session'] + '_' + df_final_long['computer_number'].astype(str)
df_final_long['belief_change'] = df_final_long['post_belief'] - df_final_long['pre_belief']


# Generate low pre-belief dummy variable
df_final_long['low_pre_belief_30'] = (df_final_long['pre_belief'] < 30).astype(int)
df_final_long['low_pre_belief_40'] = (df_final_long['pre_belief'] < 40).astype(int)
df_final_long['low_pre_belief_50'] = (df_final_long['pre_belief'] < 50).astype(int)

# save the final dataframe to an Excel file
df_final_long.to_excel('output/final_data_long.xlsx', index=False)



#######################################################################################################################

# ANALYSIS:
df_final_long = pd.read_excel("output/final_data_long.xlsx")



# SCATTERPLOT: CHANGE IN BELIEF AND INITIAL BELIEF
# Set up the plot
plt.figure(figsize=(10, 8))

# Create scatterplot
sns.scatterplot(data=df_final_long, x='pre_belief', y='belief_change', hue='treatment',
                style='treatment', palette={'STATIC': 'blue', 'CHATBOT': 'red'})

# Add horizontal line at y=0
plt.axhline(y=0, color='k', linestyle='--', alpha=0.75)

# Customize the plot
plt.xlabel('Initial Belief', fontsize=14)
plt.ylabel('Belief change', fontsize=14)

# Add legend
plt.legend(loc='best', fontsize=14, title_fontsize=14)

# Set y-axis limits (adjust as needed based on your data)
y_min, y_max = plt.ylim()
plt.ylim(y_min, y_max * 1.1)  # Extend y-axis slightly for annotations

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().set_axisbelow(True)  # Ensure grid is drawn behind the points

# Show the plot
plt.tight_layout()

# Save the figure
plt.savefig('output/scatterplot_belief_change.png', dpi=300)
plt.close()

# INTERACTION EFFECT: BELIEF CHANGE AND LOW PRE-BELIEF
import statsmodels.api as sm
import numpy as np
import pandas as pd
import math

# LOAD DATA
df_final_long = pd.read_excel("output/final_data_long.xlsx")



def run_interaction_regression(df_final_long, interaction_term):
    df_regression = df_final_long.dropna(subset=['pre_belief', 'post_belief']).copy()
    df_regression.loc[:, 'treatment_dummy'] = df_regression['treatment'].apply(lambda x: 1 if x == 'CHATBOT' else 0)
    df_regression.loc[:, interaction_term] = df_regression['treatment_dummy'] * df_regression[f'low_pre_belief_{interaction_term[-2:]}']
    df_regression.loc[:, 'belief_change'] = df_regression['post_belief'] - df_regression['pre_belief']

    X = sm.add_constant(df_regression[['treatment_dummy', f'low_pre_belief_{interaction_term[-2:]}', interaction_term, 'Q2', 'Q3', 'Q4', 'Q5']])
    y = df_regression['belief_change']

    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_regression['participant_id']})
    return model, len(df_regression)

# MAIN REGRESSION USING INDIVIDUAL BELIEF CHANGE
def run_main_regression(df_final_long):
    df_regression = df_final_long.dropna(subset=['pre_belief', 'post_belief']).copy()
    df_regression.loc[:, 'treatment_dummy'] = df_regression['treatment'].apply(lambda x: 1 if x == 'CHATBOT' else 0)
    df_regression.loc[:, 'belief_change'] = df_regression['post_belief'] - df_regression['pre_belief']

    X = sm.add_constant(df_regression[['treatment_dummy', 'Q2', 'Q3', 'Q4', 'Q5']])
    y = df_regression['belief_change']

    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_regression['participant_id']})
    return model, len(df_regression)

# Run regressions
interaction_model_30, obs_30 = run_interaction_regression(df_final_long, 'interaction_30')
interaction_model_40, obs_40 = run_interaction_regression(df_final_long, 'interaction_40')
interaction_model_50, obs_50 = run_interaction_regression(df_final_long, 'interaction_50')
main_model, obs_main = run_main_regression(df_final_long)


# Print the results
from statsmodels.iolib.summary2 import summary_col

# Combine results
results = summary_col(
    [interaction_model_50, interaction_model_40, interaction_model_30, main_model],
    model_names=[
        "IB50",
        "IB40",
        "IB30",
        "Main Model",
    ],
    stars=True,
    float_format="%0.3f",
    info_dict={'R-squared': lambda x: "{:.3f}".format(x.rsquared),
               'Adj. R-squared': lambda x: "{:.3f}".format(x.rsquared_adj),
               'Observations': lambda x: "{0:d}".format(int(x.nobs))}
)

# Convert the summary_col results to a DataFrame
df_results = results.tables[0]

# Extract and reorder the necessary parts manually
columns_order = [
    ('const', 'Constant'),
    ('treatment_dummy', 'CHATBOT'),
    ('low_pre_belief_50', 'IB50'),
    ('interaction_50', 'IB50 $\cdot$ CHATBOT'),
    ('low_pre_belief_40', 'IB40'),
    ('interaction_40', 'IB40 $\cdot$ CHATBOT'),
    ('low_pre_belief_30', 'IB30'),
    ('interaction_30', 'IB30 $\cdot$ CHATBOT')
]

# Extract the standard errors from the model results
std_errors = {
    'IB50': interaction_model_50.bse,
    'IB40': interaction_model_40.bse,
    'IB30': interaction_model_30.bse,
    'Main Model': main_model.bse
}

# Extract R-squared and Adj. R-squared values
r_squared = {
    'IB50': "{:.3f}".format(interaction_model_50.rsquared),
    'IB40': "{:.3f}".format(interaction_model_40.rsquared),
    'IB30': "{:.3f}".format(interaction_model_30.rsquared),
    'Main Model': "{:.3f}".format(main_model.rsquared)
}

# Manually construct the LaTeX table
latex_table = r"""
\begin{table}
\caption{}
\label{}
\begin{center}
\begin{tabular}{lllll}
        \hline
        Dep. var.: & \multicolumn{4}{c}{Belief change on recommended option} \\
        & \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)} & \multicolumn{1}{c}{(3)} & \multicolumn{1}{c}{(4)} \\
        \hline
"""

# Iterate over the rows and add them to the table
for var, var_name in columns_order:
    if var in df_results.index:
        coef_row = f"{var_name} & {df_results['IB50'].get(var, '')} & {df_results['IB40'].get(var, '')} & {df_results['IB30'].get(var, '')} & {df_results['Main Model'].get(var, '')} \\\\\n"
        latex_table += coef_row
        if var not in ['R-squared', 'R-squared Adj.']:
            for model in ['IB50', 'IB40', 'IB30', 'Main Model']:
                std_err_val = std_errors[model].get(var, np.nan)
                if pd.isna(std_err_val):
                    std_err_val = ''
                else:
                    std_err_val = f'({std_err_val:.3f})'
                std_err_row = f" & {std_err_val}"
                latex_table += std_err_row
            latex_table += " \\\\\n"

latex_table += r"""
\hline
"""

# Add the R-squared
latex_table += f"R-squared & {r_squared['IB50']} & {r_squared['IB40']} & {r_squared['IB30']} & {r_squared['Main Model']} \\\\\n"

# Add the number of observations and clustered SE rows
latex_table += f"Observations & {math.ceil(obs_50)} & {math.ceil(obs_40)} & {math.ceil(obs_30)} & {math.ceil(obs_main)} \\\\\n"
latex_table += "Question FEs & Yes & Yes & Yes & Yes \\\\\n"
latex_table += "Clustered SE & Yes & Yes & Yes & Yes \\\\\n"

latex_table += r"""
\hline
\end{tabular}
\end{center}
\end{table}
\bigskip
Standard errors in parentheses. \newline 
* p$<$.1, ** p$<$.05, ***p$<$.01
"""

# Print and save the LaTeX table
print(latex_table)

# Save results to a .tex file
with open('output/regression_summary_main.tex', 'w') as f:
    f.write(latex_table)





# full table
columns_order = [
    ('const', 'Constant'),
    ('treatment_dummy', 'CHATBOT'),
    ('low_pre_belief_50', 'IB50'),
    ('interaction_50', 'IB50 $\cdot$ CHATBOT'),
    ('low_pre_belief_40', 'IB40'),
    ('interaction_40', 'IB40 $\cdot$ CHATBOT'),
    ('low_pre_belief_30', 'IB30'),
    ('interaction_30', 'IB30 $\cdot$ CHATBOT'),
    ('Q2', 'Q2'),
    ('Q3', 'Q3'),
    ('Q4', 'Q4'),
    ('Q5', 'Q5')
]

# Extract the standard errors from the model results
std_errors = {
    'IB50': interaction_model_50.bse,
    'IB40': interaction_model_40.bse,
    'IB30': interaction_model_30.bse,
    'Main Model': main_model.bse
}

# Extract R-squared and Adj. R-squared values
r_squared = {
    'IB50': "{:.3f}".format(interaction_model_50.rsquared),
    'IB40': "{:.3f}".format(interaction_model_40.rsquared),
    'IB30': "{:.3f}".format(interaction_model_30.rsquared),
    'Main Model': "{:.3f}".format(main_model.rsquared)
}

# Manually construct the LaTeX table
latex_table = r"""
\begin{table}
\caption{}
\label{}
\begin{center}
\begin{tabular}{lllll}
        \hline
        Dep. var.: & \multicolumn{4}{c}{Belief change on recommended option} \\
        & \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)} & \multicolumn{1}{c}{(3)} & \multicolumn{1}{c}{(4)} \\
        \hline
"""

# Iterate over the rows and add them to the table
for var, var_name in columns_order:
    if var in df_results.index:
        coef_row = f"{var_name} & {df_results['IB50'].get(var, '')} & {df_results['IB40'].get(var, '')} & {df_results['IB30'].get(var, '')} & {df_results['Main Model'].get(var, '')} \\\\\n"
        latex_table += coef_row
        if var not in ['R-squared', 'R-squared Adj.']:
            for model in ['IB50', 'IB40', 'IB30', 'Main Model']:
                std_err_val = std_errors[model].get(var, np.nan)
                if pd.isna(std_err_val):
                    std_err_val = ''
                else:
                    std_err_val = f'({std_err_val:.3f})'
                std_err_row = f" & {std_err_val}"
                latex_table += std_err_row
            latex_table += " \\\\\n"

latex_table += r"""
\hline
"""

# Add the R-squared
latex_table += f"R-squared & {r_squared['IB50']} & {r_squared['IB40']} & {r_squared['IB30']} & {r_squared['Main Model']} \\\\\n"

# Add the number of observations and clustered SE rows
latex_table += f"Observations & {math.ceil(obs_50)} & {math.ceil(obs_40)} & {math.ceil(obs_30)} & {math.ceil(obs_main)} \\\\\n"
latex_table += "Clustered SE & Yes & Yes & Yes & Yes \\\\\n"

latex_table += r"""
\hline
\end{tabular}
\end{center}
\end{table}
\bigskip
Standard errors in parentheses. \newline 
* p$<$.1, ** p$<$.05, ***p$<$.01
"""

# Print and save the LaTeX table
print(latex_table)

# Save results to a .tex file
with open('output/regression_summary_appendix.tex', 'w') as f:
    f.write(latex_table)





# BELIEF CHANGE FOLLOWING A CORRECT / INCORRECT RECOMMENDATION

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plot
fig, axs = plt.subplots(1, 2, figsize=(20, 8))

# Filter data
df_correct = df_final_long.loc[df_final_long['rec_correct'] == True].dropna(
    subset=['pre_belief', 'post_belief', 'belief_change'])
df_incorrect = df_final_long.loc[df_final_long['rec_correct'] == False].dropna(
    subset=['pre_belief', 'post_belief', 'belief_change'])

dataframes = [df_correct, df_incorrect]
session_types = ['STATIC', 'CHATBOT']
titles = ["Correct Recommendations", "Incorrect Recommendations"]

for i, data in enumerate(dataframes):
    # Calculate mean and standard error for each session type
    grouped_data = data.groupby('treatment')['belief_change']
    mean_values = grouped_data.mean()
    stderr_values = grouped_data.sem()

    # Calculate 95% confidence intervals
    n = grouped_data.count()
    conf_intervals = stderr_values * stats.t.ppf((1 + 0.95) / 2., n - 1)

    y_pos = np.arange(len(session_types))
    colors = ['blue', 'red']  # Blue for STATIC, Red for CHATBOT

    # Create bars with error bars
    bars = axs[i].bar(y_pos, [mean_values[s] for s in session_types], yerr=[conf_intervals[s] for s in session_types],
                      align='center', alpha=0.5, capsize=10, color=colors)

    # Add labels and title
    axs[i].set_xticks(y_pos)
    axs[i].set_xticklabels(session_types, fontsize=14)
    axs[i].set_ylabel('Average belief change', fontsize=14)
    axs[i].set_title(titles[i], fontsize=14)

    # Set y-axis limits and grid
    y_max = max(mean_values + 2 * conf_intervals) * 1.1
    axs[i].set_ylim(0, y_max)
    axs[i].grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines

    # Add mean text on bars
    for j, bar in enumerate(bars):
        yval = bar.get_height()
        yerr = conf_intervals[session_types[j]]
        axs[i].text(bar.get_x() + bar.get_width() / 2, yval + yerr + 0.02 * y_max,
                    f'Mean: {mean_values[session_types[j]]:.2f}', ha='center', va='bottom', fontsize=14)

plt.tight_layout()

# Save the figure
plt.savefig('output/barchart_correct_incorrect.png', dpi=300)
plt.close()

print("Main result figure has been saved as 'barchart_correct_incorrect.png' in the output folder.")



# Perform clustered t-test
def clustered_ttest(data, value_col, cluster_col):
    # Calculate cluster means
    cluster_means = data.groupby(cluster_col)[value_col].mean()

    # Calculate overall mean
    mean_val = cluster_means.mean()

    # Perform t-test on cluster means
    t_statistic, p_value = stats.ttest_1samp(cluster_means, 0)

    return t_statistic, p_value, len(cluster_means), mean_val


t_statistic, p_value, n_clusters, mean_val = clustered_ttest(df_incorrect, 'belief_change', 'participant_id')

print(f"Clustered t-test results:")
print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")
print(f"Number of clusters (participants): {n_clusters}")
print(f"Mean value of belief change: {mean_val}")





# COMBINE WITH SURVEY DATA FOR MORE CONTROL AT INDIVIDUAL LEVEL:
df_survey = pd.read_excel("output/survey_data.xlsx")

# Merge survey data with the final long dataframe
df_final_long_survey = pd.merge(df_final_long, df_survey, how='inner', on=['computer_number', 'session'])
sex_dummies = pd.get_dummies(df_final_long_survey['survey_demographics.1.player.sex'], drop_first=True).astype(int)
sex_dummies.columns = ['Woman', 'Not_to_Say', 'Other']
df_final_long_survey = pd.concat([df_final_long_survey, sex_dummies], axis=1)
df_final_long_survey.to_excel('output/final_data_long_survey.xlsx', index=False)


# Load the merged dataset
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import math

# INTERACTION EFFECT: BELIEF CHANGE AND LOW PRE-BELIEF (MODIFIED)
def run_interaction_regression_robust(df, interaction_term):
    df_regression = df.dropna(subset=['pre_belief', 'post_belief']).copy()
    df_regression.loc[:, 'treatment_dummy'] = df_regression['treatment'].apply(lambda x: 1 if x == 'CHATBOT' else 0)
    df_regression.loc[:, interaction_term] = df_regression['treatment_dummy'] * df_regression[f'low_pre_belief_{interaction_term[-2:]}']
    df_regression.loc[:, 'belief_change'] = df_regression['post_belief'] - df_regression['pre_belief']

    features_to_include = ['treatment_dummy', f'low_pre_belief_{interaction_term[-2:]}', interaction_term, 'Q2', 'Q3', 'Q4', 'Q5', 'Woman', 'Not_to_Say', 'Other']
    X = sm.add_constant(df_regression[features_to_include])
    y = df_regression['belief_change']

    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_regression['participant_id']})
    return model, len(df_regression)

# MAIN REGRESSION USING INDIVIDUAL BELIEF CHANGE + SURVEY DATA
def run_main_regression_robust(df):
    df_regression = df.dropna(subset=['pre_belief', 'post_belief']).copy()
    df_regression.loc[:, 'treatment_dummy'] = df_regression['treatment'].apply(lambda x: 1 if x == 'CHATBOT' else 0)
    df_regression.loc[:, 'belief_change'] = df_regression['post_belief'] - df_regression['pre_belief']

    features_to_include = ['treatment_dummy', 'Q2', 'Q3', 'Q4', 'Q5', 'Woman', 'Not_to_Say', 'Other']
    X = sm.add_constant(df_regression[features_to_include])
    y = df_regression['belief_change']

    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_regression['participant_id']})
    return model, len(df_regression)

# Run regressions
interaction_model_30, obs_30 = run_interaction_regression_robust(df_final_long_survey, 'interaction_30')
interaction_model_40, obs_40 = run_interaction_regression_robust(df_final_long_survey, 'interaction_40')
interaction_model_50, obs_50 = run_interaction_regression_robust(df_final_long_survey, 'interaction_50')
main_model, obs_main = run_main_regression_robust(df_final_long_survey)

# Combine results
results = summary_col(
    [interaction_model_50, interaction_model_40, interaction_model_30, main_model],
    model_names=[
        "IB50",
        "IB40",
        "IB30",
        "Main Model",
    ],
    stars=True,
    float_format="%0.3f",
    info_dict={'R-squared': lambda x: "{:.3f}".format(x.rsquared),
               'Adj. R-squared': lambda x: "{:.3f}".format(x.rsquared_adj),
               'Observations': lambda x: "{0:d}".format(int(x.nobs))}
)

# Convert the summary_col results to a DataFrame
df_results = results.tables[0]

# Extract and reorder the necessary parts manually
columns_order = [
    ('const', 'Constant'),
    ('treatment_dummy', 'CHATBOT'),
    ('low_pre_belief_50', 'IB50'),
    ('interaction_50', 'IB50 $\cdot$ CHATBOT'),
    ('low_pre_belief_40', 'IB40'),
    ('interaction_40', 'IB40 $\cdot$ CHATBOT'),
    ('low_pre_belief_30', 'IB30'),
    ('interaction_30', 'IB30 $\cdot$ CHATBOT'),
    ('Woman', 'Woman'),
    ('Not_to_Say', 'Not to Say'),
    ('Other', 'Other')
]

# Extract the standard errors from the model results
std_errors = {
    'IB50': interaction_model_50.bse,
    'IB40': interaction_model_40.bse,
    'IB30': interaction_model_30.bse,
    'Main Model': main_model.bse
}

# Extract R-squared and Adj. R-squared values
r_squared = {
    'IB50': "{:.3f}".format(interaction_model_50.rsquared),
    'IB40': "{:.3f}".format(interaction_model_40.rsquared),
    'IB30': "{:.3f}".format(interaction_model_30.rsquared),
    'Main Model': "{:.3f}".format(main_model.rsquared)
}

# Manually construct the LaTeX table
latex_table = r"""
\begin{table}
\caption{}
\label{}
\begin{center}
\begin{tabular}{lllll}
\hline
Dep. var.: & \multicolumn{4}{c}{Belief change on recommended option} \\
& \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)} & \multicolumn{1}{c}{(3)} & \multicolumn{1}{c}{(4)} \\
\hline
"""

# Iterate over the rows and add them to the table
for var, var_name in columns_order:
    if var in df_results.index:
        coef_row = f"{var_name} & {df_results['IB50'].get(var, '')} & {df_results['IB40'].get(var, '')} & {df_results['IB30'].get(var, '')} & {df_results['Main Model'].get(var, '')} \\\\\n"
        latex_table += coef_row
        if var not in ['R-squared', 'R-squared Adj.']:
            for model in ['IB50', 'IB40', 'IB30', 'Main Model']:
                std_err_val = std_errors[model].get(var, np.nan)
                if pd.isna(std_err_val):
                    std_err_val = ''
                else:
                    std_err_val = f'({std_err_val:.3f})'
                std_err_row = f" & {std_err_val}"
                latex_table += std_err_row
            latex_table += " \\\\\n"

latex_table += r"""
\hline
"""

# Add the R-squared
latex_table += f"R-squared & {r_squared['IB50']} & {r_squared['IB40']} & {r_squared['IB30']} & {r_squared['Main Model']} \\\\\n"

# Add the number of observations and clustered SE rows
latex_table += f"Observations & {math.ceil(obs_50)} & {math.ceil(obs_40)} & {math.ceil(obs_30)} & {math.ceil(obs_main)} \\\\\n"
latex_table += "Question FEs & Yes & Yes & Yes & Yes \\\\\n"
latex_table += "Clustered SE & Yes & Yes & Yes & Yes \\\\\n"

latex_table += r"""
\hline
\end{tabular}
\end{center}
\end{table}
\bigskip
Standard errors in parentheses. \newline 
* p$<$.1, ** p$<$.05, ***p$<$.01
"""

# Print and save the LaTeX table
print(latex_table)

# Save results to a .tex file
with open('output/regression_summary_robust.tex', 'w') as f:
    f.write(latex_table)





# Additional statistics report:
# Number of proper recommendations:
condition = (df_final_long['rec_label'].isin(['A', 'B', 'C', 'D'])) & (df_final_long['representative'] == 1) & (df_final_long['treatment'] == 'CHATBOT')
condition = (df_final_long['rec_label'].isin(['Z', 'I', ])) & (df_final_long['treatment'] == 'CHATBOT')
count = df_final_long[condition].shape[0]
total = df_final_long[df_final_long['treatment'] == 'CHATBOT'].shape[0]
print(count)



# EXPLORATORY: ABSOLUTE BELIEF CHANGE #
df_regression = df_final_long.dropna(subset=['pre_belief', 'post_belief']).copy()

# Use .loc accessor for setting values
df_regression.loc[:, 'treatment_dummy'] = df_regression['treatment'].apply(lambda x: 1 if x == 'CHATBOT' else 0)

X = sm.add_constant(df_regression[['treatment_dummy', 'Q2', 'Q3', 'Q4', 'Q5']])
y = df_regression['belief_change_absolute']

model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_regression['participant_id']})

print(model.summary())








