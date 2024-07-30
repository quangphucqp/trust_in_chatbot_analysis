import pandas as pd
import numpy as np
from scipy import stats

df_main = pd.read_excel("output/final_data_unshuffled.xlsx")
# Drop the rows that have no usable label:
# Create a boolean mask for each rec_label column
masks = [df_main[f'rec_label.{i}'].isin(['A', 'B', 'C', 'D']) for i in range(1, 6)]

# Combine the masks using the OR operator
combined_mask = pd.concat(masks, axis=1).any(axis=1)

# Filter the DataFrame to DISCARD the rows where none of the conditions are met
df_main = df_main[combined_mask]
df_main['treatment'] = df_main['session'].apply(lambda x: 'CHATBOT' if 'X_session' in x else 'STATIC')



# Create a dictionary to store the labels for each categorical column
labels = {
    'survey_demographics.1.player.sex': {
        'label': 'Are you a:',
        'labels': {
            1: 'Man',
            2: 'Woman',
            3: 'Prefer not to say',
            4: 'Other (you can indicate in next page)'
        }
    },
    'survey_demographics.1.player.home_country': {
        'label': 'Where is your home country?',
        'labels': {
            1: 'Netherlands',
            2: 'Europe (other than the Netherlands)',
            3: 'Africa',
            4: 'USA/Canada',
            5: 'Latin/South America',
            6: 'West/Central Asia',
            7: 'East/South East Asia & South Asia',
            8: 'Other'
        }
    },
    'survey_demographics.1.player.field_of_study': {
        'label': 'What is your field of study?',
        'labels': {
            1: 'Economics and Management',
            2: 'Law',
            3: 'Social and Behavioral Sciences',
            4: 'Theology',
            5: 'Humanities',
            6: 'Other'
        }
    },
    'survey_demographics.1.player.study_program': {
        'label': 'Which program do you follow?',
        'labels': {
            1: 'Bachelor',
            2: 'Master',
            3: 'PhD',
            4: 'Other'
        }
    },
    'survey_comprehensive.1.player.trust_people': {
        'label': 'Generally speaking, would you say that most people can be trusted or that you can\'t be too careful in dealing with people?',
        'labels': {
            1: 'Most people can be trusted',
            0: 'Can\'t be too careful'
        }
    },
    'survey_comprehensive.1.player.use_ai_assistants': {
        'label': 'How often do you use AI assistants (such as Siri, Alexa, Google Assistant) in your daily life?',
        'labels': {
            1: 'Never',
            2: 'Rarely',
            3: 'Sometimes',
            4: 'Often',
            5: 'Daily'
        }
    },
    'survey_comprehensive.1.player.interact_ai_chatbots': {
        'label': 'How frequently do you interact with AI-powered chatbots, such as ChatGPT?',
        'labels': {
            1: 'Never',
            2: 'Rarely',
            3: 'Sometimes',
            4: 'Often',
            5: 'Daily'
        }
    },
    'survey_comprehensive.1.player.pay_for_ai': {
        'label': 'Do you currently pay for or subscribe to any premium versions of AI assistants or chatbots?',
        'labels': {
            True: 'Yes',
            False: 'No'
        }
    },
    'survey_comprehensive.1.player.ai_experience_level': {
        'label': 'How would you describe your experience level with AI assistants and chatbots?',
        'labels': {
            1: 'Novice (little to no experience)',
            2: 'Beginner (some basic experience)',
            3: 'Intermediate (moderate experience)',
            4: 'Advanced (extensive experience)'
        }
    }
}



# Export survey data:
# List of all survey variables
survey_vars = [
    'survey_demographics.1.player.age',
    'survey_demographics.1.player.sex',
    'survey_demographics.1.player.home_country',
    'survey_demographics.1.player.field_of_study',
    'survey_demographics.1.player.study_program',
    'survey_comprehensive.1.player.risk_aversion',
    'survey_comprehensive.1.player.correct_answers_estimate',
    'survey_comprehensive.1.player.performance_percentile',
    'survey_comprehensive.1.player.logical_ability',
    'survey_comprehensive.1.player.trust_people',
    'survey_comprehensive.1.player.fair_people',
    'survey_comprehensive.1.player.helpful_people',
    'survey_comprehensive.1.player.use_ai_assistants',
    'survey_comprehensive.1.player.interact_ai_chatbots',
    'survey_comprehensive.1.player.pay_for_ai',
    'survey_comprehensive.1.player.ai_experience_level',
    'survey_comprehensive.1.player.ai_skill_comparison'
]

# Add 'computer_number' and 'session' to the list of columns to save
columns_to_save = ['computer_number', 'session'] + survey_vars

# Create a new DataFrame with only the selected columns
df_survey = df_main[columns_to_save].copy()

# Save the new DataFrame to an Excel file
output_file = 'output/survey_data.xlsx'
df_survey.to_excel(output_file, index=False)


# Function to apply labels to a column
def apply_labels(column, label_info):
    return column.map(label_info['labels'])


# Function to calculate descriptive statistics for categorical variables by session type
def categorical_descriptives_by_session(df, var, labels):
    stats = []
    for treatment in ['X', 'O']:
        session_df = df[df['treatment'] == treatment]
        counts = session_df[var].value_counts()
        percentages = session_df[var].value_counts(normalize=True) * 100

        for category in counts.index:
            stats.append({
                'Session Type': treatment,
                'Category': labels.get(category, category),
                'Count': counts[category],
                'Percentage': round(percentages[category], 2)
            })

    return pd.DataFrame(stats)


# Function to calculate descriptive statistics for numerical variables by session type
def numerical_descriptives_by_session(df, var):
    stats = []
    for treatment in ['X', 'O']:
        session_df = df[df['treatment'] == treatment]
        desc = session_df[var].describe()
        stats.append({
            'Session Type': treatment,
            'Count': desc['count'],
            'Mean': round(desc['mean'], 2),
            'Std': round(desc['std'], 2),
            'Min': desc['min'],
            '25%': desc['25%'],
            '50%': desc['50%'],
            '75%': desc['75%'],
            'Max': desc['max']
        })

    return pd.DataFrame(stats)


# Apply the labels to the DataFrame
for column, label_info in labels.items():
    if column in df_main.columns:
        df_main[f'{column}_labeled'] = apply_labels(df_main[column], label_info)






#### DEMOGRAPHICS ####
import pandas as pd
import numpy as np


# List of demographic variables
demographic_vars = [
    'survey_demographics.1.player.age',
    'survey_demographics.1.player.sex',
    'survey_demographics.1.player.home_country',
    'survey_demographics.1.player.field_of_study',
    'survey_demographics.1.player.study_program'
]



# Dictionary to store results
results = {}

# Generate descriptive statistics for each variable
for var in demographic_vars:
    if var in df_main.columns:
        if var in labels and 'labels' in labels[var]:
            # Categorical variables
            label_dict = labels[var]['labels']
            results[var] = categorical_descriptives_by_session(df_main, var, label_dict)
        else:
            # Numerical variables
            results[var] = numerical_descriptives_by_session(df_main, var)

# Print results in a nicely formatted way
for var, stats in results.items():
    print(f"\nDescriptive Statistics for {labels.get(var, {}).get('label', var)} by Session Type")
    print("-" * 80)
    print(stats.to_string(index=False))
    print("\n")


#### COMPREHENSIVE SURVEY ####


# List of comprehensive survey variables
comprehensive_vars = [
    'survey_comprehensive.1.player.risk_aversion',
    'survey_comprehensive.1.player.correct_answers_estimate',
    'survey_comprehensive.1.player.performance_percentile',
    'survey_comprehensive.1.player.logical_ability',
    'survey_comprehensive.1.player.trust_people',
    'survey_comprehensive.1.player.fair_people',
    'survey_comprehensive.1.player.helpful_people',
    'survey_comprehensive.1.player.use_ai_assistants',
    'survey_comprehensive.1.player.interact_ai_chatbots',
    'survey_comprehensive.1.player.pay_for_ai',
    'survey_comprehensive.1.player.ai_experience_level',
    'survey_comprehensive.1.player.ai_skill_comparison'
]


# Function to calculate descriptive statistics for categorical variables
def categorical_descriptives(series, labels):
    counts = series.value_counts()
    percentages = series.value_counts(normalize=True) * 100

    stats = pd.DataFrame({
        'Category': [labels.get(k, k) for k in counts.index],
        'Count': counts.values,
        'Percentage': percentages.values
    })
    stats['Percentage'] = stats['Percentage'].round(2)
    return stats.sort_index()


# Dictionary to store results
results = {}

# Generate descriptive statistics for each variable
for var in comprehensive_vars:
    if var in df_main.columns:
        if var in labels and 'labels' in labels[var]:
            # Categorical variables
            label_dict = labels[var]['labels']
            results[var] = categorical_descriptives(df_main[var], label_dict)
        else:
            # Numerical variables
            results[var] = df_main[var].describe()

# Print results in a nicely formatted way
for var, stats in results.items():
    print(f"\nDescriptive Statistics for {labels.get(var, {}).get('label', var)}")
    print("-" * 60)
    if isinstance(stats, pd.Series):  # Numeric statistics
        print(stats.to_string())
    else:  # Categorical statistics
        print(stats.to_string(index=False))
    print("\n")






# BALANCE TABLE:
import pandas as pd
from scipy import stats



# List of all demographic variables
demographic_vars = [
    'survey_demographics.1.player.age',
    'survey_demographics.1.player.sex',
    'survey_demographics.1.player.home_country',
    'survey_demographics.1.player.field_of_study',
    'survey_demographics.1.player.study_program'
]

# Label dictionaries
labels = {
    'survey_demographics.1.player.sex': {
        'label': 'Are you a:',
        'labels': {
            1: 'Man',
            2: 'Woman',
            3: 'Prefer not to say',
            4: 'Other (you can indicate in next page)'
        }
    },
    'survey_demographics.1.player.home_country': {
        'label': 'Where is your home country?',
        'labels': {
            1: 'Netherlands',
            2: 'Europe (other than the Netherlands)',
            3: 'Africa',
            4: 'USA/Canada',
            5: 'Latin/South America',
            6: 'West/Central Asia',
            7: 'East/South East Asia & South Asia',
            8: 'Other'
        }
    },
    'survey_demographics.1.player.field_of_study': {
        'label': 'What is your field of study?',
        'labels': {
            1: 'Economics and Management',
            2: 'Law',
            3: 'Social and Behavioral Sciences',
            4: 'Theology',
            5: 'Humanities',
            6: 'Other'
        }
    },
    'survey_demographics.1.player.study_program': {
        'label': 'Which program do you follow?',
        'labels': {
            1: 'Bachelor',
            2: 'Master',
            3: 'PhD',
            4: 'Other'
        }
    }
}

def generate_treatment_stats(df, variables, labels):
    results = []
    for var in variables:
        if var in df.columns:
            # Descriptive statistics
            desc_stats = df.groupby('treatment')[var].agg(['mean', 'std', 'count'])

            # Check if the variable is categorical or numerical
            if var in labels:  # For categorical variables with provided labels
                contingency_table = pd.crosstab(df['treatment'], df[var])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                test_name = 'Chi-square'
                test_statistic = chi2

                # Get fraction representation
                fraction_representation = contingency_table.div(contingency_table.sum(1), axis=0)
                chatbot_fractions = ", ".join([f"{labels[var]['labels'][idx]}: {frac:.2f}" for idx, frac in fraction_representation.loc['CHATBOT'].items()])
                static_fractions = ", ".join([f"{labels[var]['labels'][idx]}: {frac:.2f}" for idx, frac in fraction_representation.loc['STATIC'].items()])

                results.append({
                    'Variable': labels[var]['label'],
                    'CHATBOT Mean (SD)': chatbot_fractions,
                    'STATIC Mean (SD)': static_fractions,
                    'CHATBOT N': int(desc_stats.loc['CHATBOT']['count']),
                    'STATIC N': int(desc_stats.loc['STATIC']['count']),
                    'Test': test_name,
                    'Statistic': f"{test_statistic:.2f}",
                    'p-value': f"{p_value:.3f}"
                })
            else:  # For numerical variables
                group1 = df[df['treatment'] == 'CHATBOT'][var].dropna()
                group2 = df[df['treatment'] == 'STATIC'][var].dropna()
                t_stat, p_value = stats.ttest_ind(group1, group2)
                test_name = 'T-test'
                test_statistic = t_stat

                # Format the results
                chatbot_stats = desc_stats.loc['CHATBOT']
                static_stats = desc_stats.loc['STATIC']

                results.append({
                    'Variable': var,
                    'CHATBOT Mean (SD)': f"{chatbot_stats['mean']:.2f} ({chatbot_stats['std']:.2f})",
                    'STATIC Mean (SD)': f"{static_stats['mean']:.2f} ({static_stats['std']:.2f})",
                    'CHATBOT N': int(chatbot_stats['count']),
                    'STATIC N': int(static_stats['count']),
                    'Test': test_name,
                    'Statistic': f"{test_statistic:.2f}",
                    'p-value': f"{p_value:.3f}"
                })

    return pd.DataFrame(results)

# Run the function
results_df = generate_treatment_stats(df_main, demographic_vars, labels)

# Print the consolidated table
print(results_df.to_string(index=False))

# Save results to Excel
results_df.to_excel('output/demographic_analysis_consolidated.xlsx', index=False)







