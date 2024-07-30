import pandas as pd
import numpy as np


# Read the beliefs.xlsx file
beliefs = pd.read_excel("output/beliefs.xlsx")
beliefs_unshuffled = pd.read_excel("output/beliefs_unshuffled.xlsx")

# Read the O_rec file
O_rec = pd.read_excel("output/O_rec.xlsx")
O_rec_unshuffled = pd.read_excel("output/O_rec_unshuffled.xlsx")

# Read the X_chatlog file
X_chatlog = pd.read_excel("output/X_chatlog.xlsx")
X_chatlog_unshuffled = pd.read_excel("output/X_chatlog_unshuffled.xlsx")

# Joining the data:
# Concatenate the O_rec and X_chatlog DataFrames
combined_data = pd.concat([O_rec, X_chatlog])
combined_data_unshuffled = pd.concat([O_rec_unshuffled, X_chatlog_unshuffled])

# Now, join beliefs with the combined dataset
df_final = pd.merge(beliefs, combined_data, how='inner', on=["session", "computer_number"])
df_final_unshuffled = pd.merge(beliefs_unshuffled, combined_data_unshuffled, how='inner', on=["session", "computer_number"])


### GENERATE VARIABLES ###
# Main outcome variable:

# Calculate the belief change, pre-belief, and post-belief for each round
for round_number in range(1, 6):
    # Create new columns for each round's belief change, pre-belief, and post-belief, initialize with NaN
    for df in [df_final, df_final_unshuffled]:
        df[f'belief_change.{round_number}'] = float('nan')
        df[f'pre_belief.{round_number}'] = float('nan')
        df[f'post_belief.{round_number}'] = float('nan')

    # Check value in rec_label column
    for label in ['a', 'b', 'c', 'd']:
        # Get mask of where rec_label equals the current label (A, B, C, or D)
        mask = df_final[f'rec_label.{round_number}'] == label.upper()
        mask_unshuffled = df_final_unshuffled[f'rec_label.{round_number}'] == label.upper()

        # Calculate pre-belief, post-belief, and belief change based on the mask
        for df, m in [(df_final, mask), (df_final_unshuffled, mask_unshuffled)]:
            df.loc[m, f'pre_belief.{round_number}'] = df.loc[m, f'questions_seq_groupmove.{round_number}.player.pre_belief_{label}']
            df.loc[m, f'post_belief.{round_number}'] = df.loc[m, f'questions_seq_groupmove.{round_number}.player.post_belief_{label}']
            df.loc[m, f'belief_change.{round_number}'] = df.loc[m, f'post_belief.{round_number}'] - df.loc[m, f'pre_belief.{round_number}']

# Calculate (on RECOMMENDED OPTION) average belief change, average pre-belief, and average post-belief
belief_change_columns = [f'belief_change.{i}' for i in range(1, 6)]
pre_belief_columns = [f'pre_belief.{i}' for i in range(1, 6)]
post_belief_columns = [f'post_belief.{i}' for i in range(1, 6)]

for df in [df_final, df_final_unshuffled]:
    df['average_belief_change'] = df[belief_change_columns].mean(axis=1)
    df['average_pre_belief'] = df[pre_belief_columns].mean(axis=1)
    df['average_post_belief'] = df[post_belief_columns].mean(axis=1)


# Generate belief change for each option and absolute belief change:
cols = ['a', 'b', 'c', 'd']
for df in [df_final, df_final_unshuffled]:
    for round_number in range(1, 6):
        for col in cols:
            pre_belief_col_name = f'questions_seq_groupmove.{round_number}.player.pre_belief_{col}'
            post_belief_col_name = f'questions_seq_groupmove.{round_number}.player.post_belief_{col}'
            belief_change_col_name = f'questions_seq_groupmove.{round_number}.player.belief_change_{col}'
            df[belief_change_col_name] = df[post_belief_col_name] - df[pre_belief_col_name]

# Calculating absolute belief change
for df in [df_final, df_final_unshuffled]:
    for round_number in range(1, 6):
        df[f'questions_seq_groupmove.{round_number}.player.belief_change_absolute'] = \
            df[f'questions_seq_groupmove.{round_number}.player.belief_change_a'].abs() \
            + df[f'questions_seq_groupmove.{round_number}.player.belief_change_b'].abs() \
            + df[f'questions_seq_groupmove.{round_number}.player.belief_change_c'].abs() \
            + df[f'questions_seq_groupmove.{round_number}.player.belief_change_d'].abs()

# Calculate average absolute belief change:
for df in [df_final, df_final_unshuffled]:
    df["average_belief_change_absolute"] = df[
        [f"questions_seq_groupmove.{round_number}.player.belief_change_absolute" for round_number in range(1, 6)]
    ].mean(axis=1)



# Save the final DataFrame to a new Excel file
df_final.to_excel('output/final_data.xlsx', index=False)
df_final_unshuffled.to_excel('output/final_data_unshuffled.xlsx', index=False)


### ROBUST DATAFRAME ###

# Robust dataframe: ignore the non-representative answers
df_robust = df_final.copy()
df_robust_unshuffled = df_final_unshuffled.copy()

# Loop through each round
for round_number in range(1, 6):
    # Define the representative and rec columns for the current round
    representative_col = f'representative.{round_number}'
    rec_col = f'rec_label.{round_number}'

    # Replace the value in rec_col with NaN if the value in representative_col is 0
    df_robust.loc[df_robust[representative_col] == 0, rec_col] = np.nan
    df_robust_unshuffled.loc[df_robust_unshuffled[representative_col] == 0, rec_col] = np.nan

for round_number in range(1, 6):
    # Create new columns for each round's belief change, pre-belief, and post-belief, initialize with NaN
    for df in [df_robust, df_robust_unshuffled]:
        df[f'belief_change.{round_number}'] = float('nan')
        df[f'pre_belief.{round_number}'] = float('nan')
        df[f'post_belief.{round_number}'] = float('nan')

    # Check value in rec_label column
    for label in ['a', 'b', 'c', 'd']:
        # Get mask of where rec_label equals the current label (A, B, C, or D)
        mask = df_robust[f'rec_label.{round_number}'] == label.upper()
        mask_unshuffled = df_robust_unshuffled[f'rec_label.{round_number}'] == label.upper()

        # Calculate pre-belief, post-belief, and belief change based on the mask
        for df, m in [(df_robust, mask), (df_robust_unshuffled, mask_unshuffled)]:
            df.loc[m, f'pre_belief.{round_number}'] = df.loc[m, f'questions_seq_groupmove.{round_number}.player.pre_belief_{label}']
            df.loc[m, f'post_belief.{round_number}'] = df.loc[m, f'questions_seq_groupmove.{round_number}.player.post_belief_{label}']
            df.loc[m, f'belief_change.{round_number}'] = df.loc[m, f'post_belief.{round_number}'] - df.loc[m, f'pre_belief.{round_number}']

# Calculate average belief change, average pre-belief, and average post-belief
belief_change_columns = [f'belief_change.{i}' for i in range(1, 6)]
pre_belief_columns = [f'pre_belief.{i}' for i in range(1, 6)]
post_belief_columns = [f'post_belief.{i}' for i in range(1, 6)]

for df in [df_robust, df_robust_unshuffled]:
    df['average_belief_change'] = df[belief_change_columns].mean(axis=1)
    df['average_pre_belief'] = df[pre_belief_columns].mean(axis=1)
    df['average_post_belief'] = df[post_belief_columns].mean(axis=1)


# Generate belief change for each option and absolute belief change:
cols = ['a', 'b', 'c', 'd']
for df in [df_robust, df_robust_unshuffled]:
    for round_number in range(1, 6):
        for col in cols:
            pre_belief_col_name = f'questions_seq_groupmove.{round_number}.player.pre_belief_{col}'
            post_belief_col_name = f'questions_seq_groupmove.{round_number}.player.post_belief_{col}'
            belief_change_col_name = f'questions_seq_groupmove.{round_number}.player.belief_change_{col}'
            df[belief_change_col_name] = df[post_belief_col_name] - df[pre_belief_col_name]

# Calculating absolute belief change
for df in [df_robust, df_robust_unshuffled]:
    for round_number in range(1, 6):
        df[f'questions_seq_groupmove.{round_number}.player.belief_change_absolute'] = \
            df[f'questions_seq_groupmove.{round_number}.player.belief_change_a'].abs() \
            + df[f'questions_seq_groupmove.{round_number}.player.belief_change_b'].abs() \
            + df[f'questions_seq_groupmove.{round_number}.player.belief_change_c'].abs() \
            + df[f'questions_seq_groupmove.{round_number}.player.belief_change_d'].abs()

# Calculate average absolute belief change:
for df in [df_robust, df_robust_unshuffled]:
    df["average_belief_change_absolute"] = df[
        [f"questions_seq_groupmove.{round_number}.player.belief_change_absolute" for round_number in range(1, 6)]
    ].mean(axis=1)


# Save the robust DataFrame to a new Excel file
df_robust.to_excel('output/robust_data.xlsx', index=False)
df_robust_unshuffled.to_excel('output/robust_data_unshuffled.xlsx', index=False)