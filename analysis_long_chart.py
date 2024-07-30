import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col

# Read the final data file
df_final_long = pd.read_excel("output/final_data_long.xlsx")

# Correctness of answers:

# Calculate the average pre_belief_correct and post_belief_correct across all questions, grouped by session_type
average_pre_belief_correct = df_final_long.groupby('treatment')['pre_belief_correct'].mean()
average_post_belief_correct = df_final_long.groupby('treatment')['post_belief_correct'].mean()

print("Average pre-belief in correct answer by treatment:")
print(average_pre_belief_correct)
print("\nAverage post-belief in correct answer by treatment:")
print(average_post_belief_correct)

# SDs:
std_dev_pre = df_final_long.groupby('treatment')['pre_belief_correct'].std()
std_dev_post = df_final_long.groupby('treatment')['post_belief_correct'].std()
print("\nStandard deviation of pre-belief in correct answer by treatment:")
print(std_dev_pre)
print("\nStandard deviation of post-belief in correct answer by treatment:")
print(std_dev_post)

# Count the number of observations in each session type
count_by_session = df_final_long.groupby('treatment')['pre_belief_correct'].count()
print("\nNumber of observations by treatment:")
print(count_by_session)

# Perform clustered t-test using OLS regression for pre_belief and post_belief
df_regression = df_final_long.dropna(subset=['pre_belief_correct', 'post_belief_correct'])
df_regression['treatment_dummy'] = df_regression['treatment'].apply(lambda x: 1 if x == 'CHATBOT' else 0)
X = sm.add_constant(df_regression['treatment_dummy'])

# Pre-belief-correct regression
y_pre = df_regression['pre_belief_correct']
model_pre = sm.OLS(y_pre, X).fit(cov_type='cluster', cov_kwds={'groups': df_regression['participant_id']})
print("\nPre-belief Regression results:")
print(model_pre.summary())

# Post-belief-correct regression
y_post = df_regression['post_belief_correct']
model_post = sm.OLS(y_post, X).fit(cov_type='cluster', cov_kwds={'groups': df_regression['participant_id']})
print("\nPost-belief Regression results:")
print(model_post.summary())



# Plot the average pre-belief and post-belief by session type
# Calculate mean pre and post belief per question and session type
belief_summary = df_final_long.groupby(['question_number', 'treatment'])[
    ['pre_belief_correct', 'post_belief_correct']].mean().reset_index()

# Check if we have data for both session types
session_types = belief_summary['treatment'].unique()
if len(session_types) != 2:
    print(f"Warning: Expected 2 session types, but found {len(session_types)}: {session_types}")
    print("Please check your data. The chart may not display as expected.")

# Set up the plot
fig, axs = plt.subplots(len(session_types), 1, figsize=(15, 10*len(session_types)), sharex=True)
sns.set_style("whitegrid")

# Create the grouped bar chart
bar_width = 0.35
questions = belief_summary['question_number'].unique()
index = np.arange(len(questions))

for i, session in enumerate(session_types):
    data = belief_summary[belief_summary['treatment'] == session]

    if len(data) == 0:
        print(f"No data for treatment {session}")
        axs[i].text(0.5, 0.5, f"No data for session type {session}", ha='center', va='center')
        continue

    axs[i].bar(index - bar_width/2, data['pre_belief_correct'], bar_width, label='Pre-belief', alpha=0.8)
    axs[i].bar(index + bar_width/2, data['post_belief_correct'], bar_width, label='Post-belief', alpha=0.6)

    # Customize each subplot
    axs[i].set_ylabel('Belief in Correct Answer', fontsize=12)
    axs[i].set_title(f'Session Type: {session}', fontsize=14)
    axs[i].set_xticks(index)
    axs[i].set_xticklabels(questions)
    axs[i].legend(loc='best')

# Customize the overall plot
plt.xlabel('Question Number', fontsize=12)
fig.suptitle('Pre and Post Belief per question and treatment', fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print(belief_summary.groupby('treatment')[['pre_belief_correct', 'post_belief_correct']].describe())


def get_treatment_means(df, outcome_var):
    # Create dummy variables for treatments
    treatment_dummies = pd.get_dummies(df['treatment'], prefix='treatment', drop_first=True)

    # Prepare the data for regression
    X = sm.add_constant(treatment_dummies)
    y = df[outcome_var]

    # Fit the model with clustered standard errors
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['participant_id']})

    # Extract results
    coef = model.params
    se = model.bse

    # Calculate means and standard errors for each treatment
    results = {}
    base_treatment = df['treatment'].unique()[0]  # Assuming the first treatment is the base (omitted) category
    results[base_treatment] = {
        'mean': coef['const'],
        'se': se['const']
    }

    for treatment in df['treatment'].unique()[1:]:
        treatment_dummy = f'treatment_{treatment}'
        results[treatment] = {
            'mean': coef['const'] + coef[treatment_dummy],
            'se': np.sqrt(
                se['const'] ** 2 + se[treatment_dummy] ** 2 + 2 * model.cov_params().loc['const', treatment_dummy])
        }

    return results, model


# Analyze pre_belief_correct
pre_results, pre_model = get_treatment_means(df_final_long, 'pre_belief_correct')

# Analyze post_belief_correct
post_results, post_model = get_treatment_means(df_final_long, 'post_belief_correct')

# Print results
print("Pre-belief Correct Means by Treatment:")
for treatment, values in pre_results.items():
    print(f"{treatment}: Mean = {values['mean']:.4f}, SE = {values['se']:.4f}")

print("\nPost-belief Correct Means by Treatment:")
for treatment, values in post_results.items():
    print(f"{treatment}: Mean = {values['mean']:.4f}, SE = {values['se']:.4f}")

# Print regression summaries
summaries = summary_col([pre_model, post_model],
                        model_names=['Pre-belief', 'Post-belief'],
                        stars=True,
                        float_format='%0.4f',
                        info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                   'R2': lambda x: "{:.4f}".format(x.rsquared)})

print("\nRegression Summaries:")
print(summaries)


