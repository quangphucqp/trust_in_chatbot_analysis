import pandas as pd
import numpy as np
import scipy
from scipy.stats import entropy
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.api as sm
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col

# Read the final data file
df_final_long_survey = pd.read_excel("output/final_data_long_survey.xlsx")
df_final_long_survey['belief_change_absolute'] = df_final_long_survey['belief_change_absolute']*0.5


# Calculate entropy and KL divergence
def calculate_entropy(df):
    # Function to get belief probabilities
    def get_probs(row, prefix):
        probs = row[[f'{prefix}_belief_a', f'{prefix}_belief_b', f'{prefix}_belief_c', f'{prefix}_belief_d']].astype(float).values / 100
        return np.clip(probs, 1e-10, 1)

    # Function to calculate entropy
    def calc_entropy(row, prefix):
        probs = get_probs(row, prefix)
        return entropy(probs)

    # Calculate entropies
    df['pre_entropy'] = df.apply(lambda row: calc_entropy(row, 'pre'), axis=1)
    df['post_entropy'] = df.apply(lambda row: calc_entropy(row, 'post'), axis=1)
    df['entropy_difference'] = df['post_entropy'] - df['pre_entropy']

    return df


# Apply the function to the dataframe
df_final_long_survey = calculate_entropy(df_final_long_survey)




# ANALYSIS:

# REGRESSIONS USING MAIN TREATMENT DUMMY AND INTERACTION EFFECTS:
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import math

# Function to run regression and return the model
def run_regression(df, y_column):
    df_regression = df.dropna(subset=['pre_belief', 'post_belief']).copy()
    df_regression.loc[:, 'treatment_dummy'] = df_regression['treatment'].apply(lambda x: 1 if x == 'CHATBOT' else 0)
    X = sm.add_constant(df_regression[['treatment_dummy', 'Q2', 'Q3', 'Q4', 'Q5']])
    y = df_regression[y_column]
    return sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_regression['participant_id']}), len(df_regression)

# Function to run regression with interaction terms
def run_interaction_regression(df, interaction_term, y_column):
    df_regression = df.dropna(subset=['pre_belief', 'post_belief']).copy()
    df_regression.loc[:, 'treatment_dummy'] = df_regression['treatment'].apply(lambda x: 1 if x == 'CHATBOT' else 0)

    # Create low pre-belief dummy
    threshold = int(interaction_term[-2:])
    df_regression.loc[:, f'low_pre_belief_{threshold}'] = (df_regression['pre_belief'] < threshold).astype(int)

    # Create interaction term
    df_regression.loc[:, interaction_term] = df_regression['treatment_dummy'] * df_regression[f'low_pre_belief_{threshold}']

    X = sm.add_constant(df_regression[['treatment_dummy', f'low_pre_belief_{threshold}', interaction_term, 'Q2', 'Q3', 'Q4', 'Q5']])
    y = df_regression[y_column]
    return sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_regression['participant_id']}), len(df_regression)

# Run regressions for belief_change_absolute
interaction_model_30_abs, obs_30_abs = run_interaction_regression(df_final_long_survey, 'interaction_30', 'belief_change_absolute')
interaction_model_40_abs, obs_40_abs = run_interaction_regression(df_final_long_survey, 'interaction_40', 'belief_change_absolute')
interaction_model_50_abs, obs_50_abs = run_interaction_regression(df_final_long_survey, 'interaction_50', 'belief_change_absolute')
main_model_abs, obs_main_abs = run_regression(df_final_long_survey, 'belief_change_absolute')

# Combine results for belief_change_absolute
results_abs = summary_col(
    [interaction_model_50_abs, interaction_model_40_abs, interaction_model_30_abs, main_model_abs],
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

# Run regressions for entropy_difference
interaction_model_30_ent, obs_30_ent = run_interaction_regression(df_final_long_survey, 'interaction_30', 'entropy_difference')
interaction_model_40_ent, obs_40_ent = run_interaction_regression(df_final_long_survey, 'interaction_40', 'entropy_difference')
interaction_model_50_ent, obs_50_ent = run_interaction_regression(df_final_long_survey, 'interaction_50', 'entropy_difference')
main_model_ent, obs_main_ent = run_regression(df_final_long_survey, 'entropy_difference')

# Combine results for entropy_difference
results_ent = summary_col(
    [interaction_model_50_ent, interaction_model_40_ent, interaction_model_30_ent, main_model_ent],
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
df_results_abs = results_abs.tables[0]
df_results_ent = results_ent.tables[0]

# Define columns order
columns_order = [
    ('const', 'Constant'),
    ('treatment_dummy', 'CHATBOT'),
    ('low_pre_belief_50', 'IB50'),
    ('interaction_50', 'IB50 $\cdot$ CHATBOT'),
    ('low_pre_belief_40', 'IB40'),
    ('interaction_40', 'IB40 $\cdot$ CHATBOT'),
    ('low_pre_belief_30', 'IB30'),
    ('interaction_30', 'IB30 $\cdot$ CHATBOT'),
]

# Extract the standard errors from the model results
std_errors_abs = {
    'IB50': interaction_model_50_abs.bse,
    'IB40': interaction_model_40_abs.bse,
    'IB30': interaction_model_30_abs.bse,
    'Main Model': main_model_abs.bse
}

std_errors_ent = {
    'IB50': interaction_model_50_ent.bse,
    'IB40': interaction_model_40_ent.bse,
    'IB30': interaction_model_30_ent.bse,
    'Main Model': main_model_ent.bse
}

# Extract R-squared values
r_squared_abs = {
    'IB50': "{:.3f}".format(interaction_model_50_abs.rsquared),
    'IB40': "{:.3f}".format(interaction_model_40_abs.rsquared),
    'IB30': "{:.3f}".format(interaction_model_30_abs.rsquared),
    'Main Model': "{:.3f}".format(main_model_abs.rsquared)
}

r_squared_ent = {
    'IB50': "{:.3f}".format(interaction_model_50_ent.rsquared),
    'IB40': "{:.3f}".format(interaction_model_40_ent.rsquared),
    'IB30': "{:.3f}".format(interaction_model_30_ent.rsquared),
    'Main Model': "{:.3f}".format(main_model_ent.rsquared)
}

# Manually construct the LaTeX table for belief_change_absolute
latex_table_abs = r"""
\begin{table}
\caption{Effect of chatbot recommendations on belief distribution}
\label{tab:regression_results_abs}
\begin{center}
\begin{tabular}{lllll}
\hline
Dep. var.: & \multicolumn{4}{c}{Variational distance in belief distribution} \\
& \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)} & \multicolumn{1}{c}{(3)} & \multicolumn{1}{c}{(4)} \\
\hline
"""

# Iterate over the rows and add them to the table
for var, var_name in columns_order:
    if var in df_results_abs.index:
        coef_row = f"{var_name} & {df_results_abs['IB50'].get(var, '')} & {df_results_abs['IB40'].get(var, '')} & {df_results_abs['IB30'].get(var, '')} & {df_results_abs['Main Model'].get(var, '')} \\\\\n"
        latex_table_abs += coef_row
        if var not in ['R-squared', 'R-squared Adj.']:
            for model in ['IB50', 'IB40', 'IB30', 'Main Model']:
                std_err_val = std_errors_abs[model].get(var, np.nan)
                if pd.isna(std_err_val):
                    std_err_val = ''
                else:
                    std_err_val = f'({std_err_val:.3f})'
                std_err_row = f" & {std_err_val}"
                latex_table_abs += std_err_row
            latex_table_abs += " \\\\\n"

latex_table_abs += r"""
\hline
"""

# Add the R-squared
latex_table_abs += f"R-squared & {r_squared_abs['IB50']} & {r_squared_abs['IB40']} & {r_squared_abs['IB30']} & {r_squared_abs['Main Model']} \\\\\n"

# Add the number of observations and clustered SE rows
latex_table_abs += f"Observations & {math.ceil(obs_50_abs)} & {math.ceil(obs_40_abs)} & {math.ceil(obs_30_abs)} & {math.ceil(obs_main_abs)} \\\\\n"
latex_table_abs += "Question FEs & Yes & Yes & Yes & Yes \\\\\n"
latex_table_abs += "Clustered SE & Yes & Yes & Yes & Yes \\\\\n"

latex_table_abs += r"""
\hline
\end{tabular}
\end{center}
\end{table}
\bigskip
Standard errors in parentheses. \newline 
* p$<$.1, ** p$<$.05, ***p$<$.01
"""

# Print and save the LaTeX table for belief_change_absolute
print(latex_table_abs)
with open('output/regression_summary_abs.tex', 'w') as f:
    f.write(latex_table_abs)


# Manually construct the LaTeX table for entropy_difference
latex_table_ent = r"""
\begin{table}
\caption{Effect of chatbot recommendation on belief distribution}
\label{tab:regression_results_ent}
\begin{center}
\begin{tabular}{lllll}
\hline
Dep. var.: & \multicolumn{4}{c}{Entropy difference in belief distribution} \\
& \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)} & \multicolumn{1}{c}{(3)} & \multicolumn{1}{c}{(4)} \\
\hline
"""

# Iterate over the rows and add them to the table
for var, var_name in columns_order:
    if var in df_results_ent.index:
        coef_row = f"{var_name} & {df_results_ent['IB50'].get(var, '')} & {df_results_ent['IB40'].get(var, '')} & {df_results_ent['IB30'].get(var, '')} & {df_results_ent['Main Model'].get(var, '')} \\\\\n"
        latex_table_ent += coef_row
        if var not in ['R-squared', 'R-squared Adj.']:
            for model in ['IB50', 'IB40', 'IB30', 'Main Model']:
                std_err_val = std_errors_ent[model].get(var, np.nan)
                if pd.isna(std_err_val):
                    std_err_val = ''
                else:
                    std_err_val = f'({std_err_val:.3f})'
                std_err_row = f" & {std_err_val}"
                latex_table_ent += std_err_row
            latex_table_ent += " \\\\\n"

latex_table_ent += r"""
\hline
"""

# Add the R-squared
latex_table_ent += f"R-squared & {r_squared_ent['IB50']} & {r_squared_ent['IB40']} & {r_squared_ent['IB30']} & {r_squared_ent['Main Model']} \\\\\n"

# Add the number of observations and clustered SE rows
latex_table_ent += f"Observations & {math.ceil(obs_50_ent)} & {math.ceil(obs_40_ent)} & {math.ceil(obs_30_ent)} & {math.ceil(obs_main_ent)} \\\\\n"
latex_table_ent += "Question FEs & Yes & Yes & Yes & Yes \\\\\n"
latex_table_ent += "Clustered SE & Yes & Yes & Yes & Yes \\\\\n"

latex_table_ent += r"""
\hline
\end{tabular}
\end{center}
\end{table}
\bigskip
Standard errors in parentheses. \newline 
* p$<$.1, ** p$<$.05, ***p$<$.01
"""

# Print and save the LaTeX table for entropy_difference
print(latex_table_ent)
with open('output/regression_summary_ent.tex', 'w') as f:
    f.write(latex_table_ent)





# FULL TABLE:

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
# Manually construct the LaTeX table for belief_change_absolute
latex_table_abs = r"""
\begin{table}
\caption{Effect of chatbot recommendations on belief distribution}
\label{tab:regression_results_abs}
\begin{center}
\begin{tabular}{lllll}
\hline
Dep. var.: & \multicolumn{4}{c}{Variational distance in belief distribution} \\
& \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)} & \multicolumn{1}{c}{(3)} & \multicolumn{1}{c}{(4)} \\
\hline
"""

# Iterate over the rows and add them to the table
for var, var_name in columns_order:
    if var in df_results_abs.index:
        coef_row = f"{var_name} & {df_results_abs['IB50'].get(var, '')} & {df_results_abs['IB40'].get(var, '')} & {df_results_abs['IB30'].get(var, '')} & {df_results_abs['Main Model'].get(var, '')} \\\\\n"
        latex_table_abs += coef_row
        if var not in ['R-squared', 'R-squared Adj.']:
            for model in ['IB50', 'IB40', 'IB30', 'Main Model']:
                std_err_val = std_errors_abs[model].get(var, np.nan)
                if pd.isna(std_err_val):
                    std_err_val = ''
                else:
                    std_err_val = f'({std_err_val:.3f})'
                std_err_row = f" & {std_err_val}"
                latex_table_abs += std_err_row
            latex_table_abs += " \\\\\n"

latex_table_abs += r"""
\hline
"""

# Add the R-squared
latex_table_abs += f"R-squared & {r_squared_abs['IB50']} & {r_squared_abs['IB40']} & {r_squared_abs['IB30']} & {r_squared_abs['Main Model']} \\\\\n"

# Add the number of observations and clustered SE rows
latex_table_abs += f"Observations & {math.ceil(obs_50_abs)} & {math.ceil(obs_40_abs)} & {math.ceil(obs_30_abs)} & {math.ceil(obs_main_abs)} \\\\\n"
latex_table_abs += "Question FEs & Yes & Yes & Yes & Yes \\\\\n"
latex_table_abs += "Clustered SE & Yes & Yes & Yes & Yes \\\\\n"

latex_table_abs += r"""
\hline
\end{tabular}
\end{center}
\end{table}
\bigskip
Standard errors in parentheses. \newline 
* p$<$.1, ** p$<$.05, ***p$<$.01
"""

# Print and save the LaTeX table for belief_change_absolute
print(latex_table_abs)
with open('output/regression_summary_abs_full.tex', 'w') as f:
    f.write(latex_table_abs)


# Manually construct the LaTeX table for entropy_difference
latex_table_ent = r"""
\begin{table}
\caption{Effect of chatbot recommendation on belief distribution}
\label{tab:regression_results_ent}
\begin{center}
\begin{tabular}{lllll}
\hline
Dep. var.: & \multicolumn{4}{c}{Entropy difference in belief distribution} \\
& \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)} & \multicolumn{1}{c}{(3)} & \multicolumn{1}{c}{(4)} \\
\hline
"""

# Iterate over the rows and add them to the table
for var, var_name in columns_order:
    if var in df_results_ent.index:
        coef_row = f"{var_name} & {df_results_ent['IB50'].get(var, '')} & {df_results_ent['IB40'].get(var, '')} & {df_results_ent['IB30'].get(var, '')} & {df_results_ent['Main Model'].get(var, '')} \\\\\n"
        latex_table_ent += coef_row
        if var not in ['R-squared', 'R-squared Adj.']:
            for model in ['IB50', 'IB40', 'IB30', 'Main Model']:
                std_err_val = std_errors_ent[model].get(var, np.nan)
                if pd.isna(std_err_val):
                    std_err_val = ''
                else:
                    std_err_val = f'({std_err_val:.3f})'
                std_err_row = f" & {std_err_val}"
                latex_table_ent += std_err_row
            latex_table_ent += " \\\\\n"

latex_table_ent += r"""
\hline
"""

# Add the R-squared
latex_table_ent += f"R-squared & {r_squared_ent['IB50']} & {r_squared_ent['IB40']} & {r_squared_ent['IB30']} & {r_squared_ent['Main Model']} \\\\\n"

# Add the number of observations and clustered SE rows
latex_table_ent += f"Observations & {math.ceil(obs_50_ent)} & {math.ceil(obs_40_ent)} & {math.ceil(obs_30_ent)} & {math.ceil(obs_main_ent)} \\\\\n"
latex_table_ent += "Question FEs & Yes & Yes & Yes & Yes \\\\\n"
latex_table_ent += "Clustered SE & Yes & Yes & Yes & Yes \\\\\n"

latex_table_ent += r"""
\hline
\end{tabular}
\end{center}
\end{table}
\bigskip
Standard errors in parentheses. \newline 
* p$<$.1, ** p$<$.05, ***p$<$.01
"""

# Print and save the LaTeX table for entropy_difference
print(latex_table_ent)
with open('output/regression_summary_ent_full.tex', 'w') as f:
    f.write(latex_table_ent)





# REGRESSION: CONTROLLING FOR SEX DUMMIES:

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import math

# Function to run regression and return the model
def run_regression(df, y_column):
    df_regression = df.dropna(subset=['pre_belief', 'post_belief']).copy()
    df_regression.loc[:, 'treatment_dummy'] = df_regression['treatment'].apply(lambda x: 1 if x == 'CHATBOT' else 0)
    features_to_include = ['treatment_dummy', 'Q2', 'Q3', 'Q4', 'Q5', 'Woman', 'Not_to_Say', 'Other']
    X = sm.add_constant(df_regression[features_to_include])
    y = df_regression[y_column]
    return sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_regression['participant_id']}), len(df_regression)

# Function to run regression with interaction terms
def run_interaction_regression(df, interaction_term, y_column):
    df_regression = df.dropna(subset=['pre_belief', 'post_belief']).copy()
    df_regression.loc[:, 'treatment_dummy'] = df_regression['treatment'].apply(lambda x: 1 if x == 'CHATBOT' else 0)

    # Create low pre-belief dummy
    threshold = int(interaction_term[-2:])
    df_regression.loc[:, f'low_pre_belief_{threshold}'] = (df_regression['pre_belief'] < threshold).astype(int)

    # Create interaction term
    df_regression.loc[:, interaction_term] = df_regression['treatment_dummy'] * df_regression[f'low_pre_belief_{threshold}']

    features_to_include = ['treatment_dummy', f'low_pre_belief_{threshold}', interaction_term, 'Q2', 'Q3', 'Q4', 'Q5', 'Woman', 'Not_to_Say', 'Other']
    X = sm.add_constant(df_regression[features_to_include])
    y = df_regression[y_column]
    return sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_regression['participant_id']}), len(df_regression)

# Run regressions for belief_change_absolute
interaction_model_30_abs, obs_30_abs = run_interaction_regression(df_final_long_survey, 'interaction_30', 'belief_change_absolute')
interaction_model_40_abs, obs_40_abs = run_interaction_regression(df_final_long_survey, 'interaction_40', 'belief_change_absolute')
interaction_model_50_abs, obs_50_abs = run_interaction_regression(df_final_long_survey, 'interaction_50', 'belief_change_absolute')
main_model_abs, obs_main_abs = run_regression(df_final_long_survey, 'belief_change_absolute')

# Combine results for belief_change_absolute
results_abs = summary_col(
    [interaction_model_50_abs, interaction_model_40_abs, interaction_model_30_abs, main_model_abs],
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

# Run regressions for entropy_difference
interaction_model_30_ent, obs_30_ent = run_interaction_regression(df_final_long_survey, 'interaction_30', 'entropy_difference')
interaction_model_40_ent, obs_40_ent = run_interaction_regression(df_final_long_survey, 'interaction_40', 'entropy_difference')
interaction_model_50_ent, obs_50_ent = run_interaction_regression(df_final_long_survey, 'interaction_50', 'entropy_difference')
main_model_ent, obs_main_ent = run_regression(df_final_long_survey, 'entropy_difference')

# Combine results for entropy_difference
results_ent = summary_col(
    [interaction_model_50_ent, interaction_model_40_ent, interaction_model_30_ent, main_model_ent],
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
df_results_abs = results_abs.tables[0]
df_results_ent = results_ent.tables[0]

# Define columns order
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
std_errors_abs = {
    'IB50': interaction_model_50_abs.bse,
    'IB40': interaction_model_40_abs.bse,
    'IB30': interaction_model_30_abs.bse,
    'Main Model': main_model_abs.bse
}

std_errors_ent = {
    'IB50': interaction_model_50_ent.bse,
    'IB40': interaction_model_40_ent.bse,
    'IB30': interaction_model_30_ent.bse,
    'Main Model': main_model_ent.bse
}

# Extract R-squared values
r_squared_abs = {
    'IB50': "{:.3f}".format(interaction_model_50_abs.rsquared),
    'IB40': "{:.3f}".format(interaction_model_40_abs.rsquared),
    'IB30': "{:.3f}".format(interaction_model_30_abs.rsquared),
    'Main Model': "{:.3f}".format(main_model_abs.rsquared)
}

r_squared_ent = {
    'IB50': "{:.3f}".format(interaction_model_50_ent.rsquared),
    'IB40': "{:.3f}".format(interaction_model_40_ent.rsquared),
    'IB30': "{:.3f}".format(interaction_model_30_ent.rsquared),
    'Main Model': "{:.3f}".format(main_model_ent.rsquared)
}

# Manually construct the LaTeX table for belief_change_absolute
latex_table_abs = r"""
\begin{table}
\caption{Robustess check: effect of chatbot recommendations on belief distribution, controlling for sample imbalance}
\label{tab:regression_results_abs}
\begin{center}
\begin{tabular}{lllll}
\hline
Dep. var.: & \multicolumn{4}{c}{Variational distance} \\
& \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)} & \multicolumn{1}{c}{(3)} & \multicolumn{1}{c}{(4)} \\
\hline
"""

# Iterate over the rows and add them to the table
for var, var_name in columns_order:
    if var in df_results_abs.index:
        coef_row = f"{var_name} & {df_results_abs['IB50'].get(var, '')} & {df_results_abs['IB40'].get(var, '')} & {df_results_abs['IB30'].get(var, '')} & {df_results_abs['Main Model'].get(var, '')} \\\\\n"
        latex_table_abs += coef_row
        if var not in ['R-squared', 'R-squared Adj.']:
            for model in ['IB50', 'IB40', 'IB30', 'Main Model']:
                std_err_val = std_errors_abs[model].get(var, np.nan)
                if pd.isna(std_err_val):
                    std_err_val = ''
                else:
                    std_err_val = f'({std_err_val:.3f})'
                std_err_row = f" & {std_err_val}"
                latex_table_abs += std_err_row
            latex_table_abs += " \\\\\n"

latex_table_abs += r"""
\hline
"""

# Add the R-squared
latex_table_abs += f"R-squared & {r_squared_abs['IB50']} & {r_squared_abs['IB40']} & {r_squared_abs['IB30']} & {r_squared_abs['Main Model']} \\\\\n"

# Add the number of observations and clustered SE rows
latex_table_abs += f"Observations & {math.ceil(obs_50_abs)} & {math.ceil(obs_40_abs)} & {math.ceil(obs_30_abs)} & {math.ceil(obs_main_abs)} \\\\\n"
latex_table_abs += "Question FEs & Yes & Yes & Yes & Yes \\\\\n"
latex_table_abs += "Clustered SE & Yes & Yes & Yes & Yes \\\\\n"

latex_table_abs += r"""
\hline
\end{tabular}
\end{center}
\end{table}
\bigskip
Standard errors in parentheses. \newline 
* p$<$.1, ** p$<$.05, ***p$<$.01
"""

# Print and save the LaTeX table for belief_change_absolute
print(latex_table_abs)
with open('output/regression_summary_abs_robust.tex', 'w') as f:
    f.write(latex_table_abs)


# Manually construct the LaTeX table for entropy_difference
latex_table_ent = r"""
\begin{table}
\caption{Robustess check: effect of chatbot recommendations on belief distribution, controlling for sample imbalance}
\label{tab:regression_results_ent}
\begin{center}
\begin{tabular}{lllll}
\hline
Dep. var.: & \multicolumn{4}{c}{Entropy difference} \\
& \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)} & \multicolumn{1}{c}{(3)} & \multicolumn{1}{c}{(4)} \\
\hline
"""

# Iterate over the rows and add them to the table
for var, var_name in columns_order:
    if var in df_results_ent.index:
        coef_row = f"{var_name} & {df_results_ent['IB50'].get(var, '')} & {df_results_ent['IB40'].get(var, '')} & {df_results_ent['IB30'].get(var, '')} & {df_results_ent['Main Model'].get(var, '')} \\\\\n"
        latex_table_ent += coef_row
        if var not in ['R-squared', 'R-squared Adj.']:
            for model in ['IB50', 'IB40', 'IB30', 'Main Model']:
                std_err_val = std_errors_ent[model].get(var, np.nan)
                if pd.isna(std_err_val):
                    std_err_val = ''
                else:
                    std_err_val = f'({std_err_val:.3f})'
                std_err_row = f" & {std_err_val}"
                latex_table_ent += std_err_row
            latex_table_ent += " \\\\\n"

latex_table_ent += r"""
\hline
"""

# Add the R-squared
latex_table_ent += f"R-squared & {r_squared_ent['IB50']} & {r_squared_ent['IB40']} & {r_squared_ent['IB30']} & {r_squared_ent['Main Model']} \\\\\n"

# Add the number of observations and clustered SE rows
latex_table_ent += f"Observations & {math.ceil(obs_50_ent)} & {math.ceil(obs_40_ent)} & {math.ceil(obs_30_ent)} & {math.ceil(obs_main_ent)} \\\\\n"
latex_table_ent += "Question FEs & Yes & Yes & Yes & Yes \\\\\n"
latex_table_ent += "Clustered SE & Yes & Yes & Yes & Yes \\\\\n"

latex_table_ent += r"""
\hline
\end{tabular}
\end{center}
\end{table}
\bigskip
Standard errors in parentheses. \newline 
* p$<$.1, ** p$<$.05, ***p$<$.01
"""

# Print and save the LaTeX table for entropy_difference
print(latex_table_ent)
with open('output/regression_summary_ent_robust.tex', 'w') as f:
    f.write(latex_table_ent)






# SCATTERPLOT: CHANGE IN BELIEF DISTRIBUTION AND PRE-BELIEF
# Set up the plot
plt.figure(figsize=(10, 8))

# Create scatterplot
sns.scatterplot(data=df_final_long_survey, x='pre_belief', y='belief_change_absolute', hue='treatment',
                style='treatment', palette={'STATIC': 'blue', 'CHATBOT': 'red'})

# Add horizontal line at y=0
plt.axhline(y=0, color='k', linestyle='--', alpha=0.75)

# Customize the plot
plt.xlabel('Pre-belief', fontsize=14)
plt.ylabel('Variational Distance', fontsize=14)

# Add legend
plt.legend(loc='best', fontsize=14, title_fontsize=14)

# Set y-axis limits (adjust as needed based on your data)
y_min, y_max = plt.ylim()
plt.ylim(y_min, y_max * 1.2)  # Extend y-axis slightly for annotations

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().set_axisbelow(True)  # Ensure grid is drawn behind the points

# Show the plot
plt.tight_layout()

# Save the figure
plt.savefig('output/scatterplot_belief_change_absolute.png', dpi=300)
plt.close()

print("Belief change scatterplot has been saved as 'scatterplot_belief_change_absolute.png' in the output folder.")



# Entropy
# Set up the plot
plt.figure(figsize=(10, 8))

# Create scatterplot
sns.scatterplot(data=df_final_long_survey, x='pre_belief', y='entropy_difference', hue='treatment',
                style='treatment', palette={'STATIC': 'blue', 'CHATBOT': 'red'})

# Add horizontal line at y=0
plt.axhline(y=0, color='k', linestyle='--', alpha=0.75)

# Customize the plot
plt.xlabel('Pre-belief', fontsize=14)
plt.ylabel('Entropy Difference', fontsize=14)

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
plt.savefig('output/scatterplot_entropy.png', dpi=300)
plt.close()

print("Belief change scatterplot has been saved as 'scatterplot_entropy.png' in the output folder.")






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# PLOTTING GENERAL CHANGE IN BELIEF DISTRIBUTION:

# Reshape data
def prepare_data(df, belief_type):
    belief_cols = [f"{belief_type}_belief_{opt}" for opt in ['a', 'b', 'c', 'd']]
    df_melted = df.melt(id_vars=['question_number', 'treatment', 'participant_id'],
                        value_vars=belief_cols,
                        var_name='option',
                        value_name='belief')
    df_melted['option'] = df_melted['option'].str[-1].str.upper()
    return df_melted

# Function to plot data
def plot_belief_distribution(df, title):
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()

    # Prepare pre and post belief data
    pre_belief_data = prepare_data(df, 'pre')
    post_belief_data = prepare_data(df, 'post')

    # Combine pre and post belief data
    pre_belief_data['belief_type'] = 'Pre'
    post_belief_data['belief_type'] = 'Post'
    combined_data = pd.concat([pre_belief_data, post_belief_data])

    base_colors = {
        'STATIC': mcolors.to_rgba('blue', alpha=0.5),  # Royal Blue #4169E1 (less intense than pure blue )
        'CHATBOT': mcolors.to_rgba('red', alpha=0.5)  # Indian Red #CD5C5C (less intense than pure red)
    }

    # Create lighter versions for pre-beliefs
    lighter_colors = {
        treatment: mcolors.to_rgba(color, alpha=0.1)
        for treatment, color in base_colors.items()
    }

    # Define the color scheme
    colors = {
        'STATIC': {'Pre': lighter_colors['STATIC'], 'Post': base_colors['STATIC']},
        'CHATBOT': {'Pre': lighter_colors['CHATBOT'], 'Post': base_colors['CHATBOT']}
    }

    bar_widths = {
        'STATIC': 0.35,  # options for the width of the bars
        'CHATBOT': 0.35
    }

    # Create subplots for each question
    for q in range(1, 6):  # Questions 1 to 5
        ax = axes[q - 1]

        mean_data = combined_data[combined_data['question_number'] == q].groupby(['option', 'belief_type', 'treatment'])[
            'belief'].mean().unstack(['belief_type', 'treatment'])

        x = np.arange(len(mean_data.index))
        for i, treatment in enumerate(['STATIC', 'CHATBOT']):
            offset = -0.2 if treatment == 'STATIC' else 0.2
            pre_values = mean_data[('Pre', treatment)]
            post_values = mean_data[('Post', treatment)]

            ax.bar(x + offset, pre_values, width=bar_widths[treatment], color=colors[treatment]['Pre'],
                   label=f"{treatment} - Initial")
            ax.bar(x + offset, post_values, width=bar_widths[treatment], color=colors[treatment]['Post'],
                   label=f"{treatment} - Updated")

            # Add arrows to show change
            for j, (pre, post) in enumerate(zip(pre_values, post_values)):
                ax.annotate('', xy=(j + offset, post), xytext=(j + offset, pre),
                            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

                # Add delta value
                delta = post - pre
                ax.text(j + offset, max(pre, post), f"{delta:+.1f}", ha='center', va='bottom', fontweight='bold', fontsize=16)

        ax.set_title(f'Question {q}', fontsize=16)
        ax.set_ylim(0, 110)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)  # Add horizontal grid lines
        ax.set_axisbelow(True)  # Ensure grid is drawn behind the bars
        ax.set_xticks(x)
        ax.set_xticklabels(mean_data.index, fontsize=16)
        ax.tick_params(axis='y', labelsize=16)

        if q == 1 or q == 4:
            ax.set_ylabel('Mean Belief', fontsize=16)
        else:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)

        if q == 5:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend([], [], frameon=False)
        else:
            ax.legend([], [], frameon=False)

    # Add legend to the unused 6th subplot
    axes[-1].axis('off')
    axes[-1].legend(handles, [
        'STATIC - Initial',
        'STATIC - Updated',
        'CHATBOT - Initial',
        'CHATBOT - Updated'
    ], loc='center', fontsize=18, title='Treatment and Belief Type', title_fontsize=18, prop={'size': 18})

    plt.tight_layout()
    fig.suptitle(title, fontsize=16, y=1.1)

# Plot for participants with initial belief larger than 50 percentage points
df_high_belief = df_final_long_survey[df_final_long_survey['low_pre_belief_50'] == 0]
plot_belief_distribution(df_high_belief, 'a) Participants with initial belief larger than 50 percentage points')
plt.savefig('output/distribution_high_IB.png')


# Plot for participants with initial beliefs lower than 50 percentage points
df_low_belief = df_final_long_survey[df_final_long_survey['low_pre_belief_50'] == 1]
plot_belief_distribution(df_low_belief, 'b) Participants with initial beliefs lower than 50 percentage points')
plt.savefig('output/distribution_low_IB.png')



