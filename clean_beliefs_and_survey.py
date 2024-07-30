import pandas as pd
import os
import numpy as np
import json


def load_files(session_name=None, data_dir_name='O'):
    base_dir = 'input'
    data_dir = os.path.join(base_dir, data_dir_name, 'beliefs')

    session_prefix = f'{data_dir_name}_session'
    file_names = [file_name for file_name in os.listdir(data_dir) if 'csv' in file_name and
                  session_prefix in file_name]

    if session_name:
        if f'{session_name}.csv' in file_names:
            file_names = [f'{session_name}.csv']
        else:
            return "The specified session does not exist."

    session_data = {}

    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)

        # Assuming that the computer number is in the filename after the '_'
        computer_number = int(''.join(filter(str.isdigit, file_name.split('_')[1])))
        df = pd.read_csv(file_path)

        session_data[computer_number] = df

    return session_data


# Append dataframes:
def append_dataframes(data_dict, session_type=None):
    df_list = []
    for key, df in data_dict.items():
        df['session'] = session_type + str(key)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# Load all files from all O sessions
all_o_session_dict = load_files(data_dir_name='O')
all_x_session_dict = load_files(data_dir_name='X')

# Append the dataframes with the session number
df_o_session = append_dataframes(all_o_session_dict, 'O_session')
df_x_session = append_dataframes(all_x_session_dict, 'X_session')


# Extract column names of both dataframes
columns_o_session = set(df_o_session.columns)
columns_x_session = set(df_x_session.columns)

# Drop the empty columns in O session
diff_columns = columns_o_session - columns_x_session
df_o_session = df_o_session.drop(diff_columns, axis=1)

# Append the X and O sessions:
df_beliefs = pd.concat([df_o_session, df_x_session], ignore_index=True)


#### PROCESSING THE BELIEFS DATA ####

cols = df_beliefs.columns
cols_to_drop = [col for col in cols if col.startswith('rec_question')]
df_beliefs = df_beliefs.drop(cols_to_drop, axis=1)


columns_to_drop = ["participant.id_in_session",
                   "participant.code",
                   "participant._is_bot",
                   "participant._index_in_pages",
                   "participant._max_page_index",
                   "participant._current_app_name",
                   "participant._current_page_name",
                   "participant.time_started_utc",
                   "participant.visited",
                   "participant.mturk_worker_id",
                   "participant.mturk_assignment_id",
                   "participant.payoff",
                   "session.code",
                   "session.label",
                   "session.mturk_HITId",
                   "session.mturk_HITGroupId",
                   "session.comment",
                   "session.is_demo",
                   "session.config.name",
                   "session.config.real_world_currency_per_point",
                   "session.config.participation_fee",
                   "consent_waitpage.1.player.id_in_group",
                   "consent_waitpage.1.player.role",
                   "consent_waitpage.1.player.payoff",
                   "consent_waitpage.1.group.id_in_subsession",
                   "consent_waitpage.1.subsession.round_number",
                   "questions_practice.1.player.id_in_group",
                   "questions_practice.1.player.role",
                   "questions_practice.1.player.payoff",
                   "end.1.player.id_in_group",
                   "end.1.player.role",
                   "end.1.player.payoff",
                   "end.1.group.id_in_subsession",
                   "end.1.subsession.round_number"]

# Use the drop() function to drop the columns
df_beliefs = df_beliefs.drop(columns_to_drop, axis=1)

df_beliefs = df_beliefs.rename(columns={'participant.label': 'computer_number'})


# Drop non-consent observations:
df_beliefs = df_beliefs[df_beliefs['consent_waitpage.1.player.consent'] != 0]


# Handle the dropped tasks where subjects cannot submit answers before time out:
# Function to process the dataframe
def replace_dropped_task_by_nan(df):
    for round_number in range(1, 6):
        pre_task_col = f'questions_seq_groupmove.{round_number}.player.pre_task_dropped'
        post_task_col = f'questions_seq_groupmove.{round_number}.player.post_task_dropped'

        if pre_task_col in df.columns:
            pre_task_dropped_rows = df[df[pre_task_col] == 1].index
            pre_belief_cols = [col for col in df.columns if
                               col.startswith(f'questions_seq_groupmove.{round_number}.player.pre_belief_')]
            df.loc[pre_task_dropped_rows, pre_belief_cols] = np.nan

        if post_task_col in df.columns:
            post_task_dropped_rows = df[df[post_task_col] == 1].index
            post_belief_cols = [col for col in df.columns if
                                col.startswith(f'questions_seq_groupmove.{round_number}.player.post_belief_')]
            df.loc[post_task_dropped_rows, post_belief_cols] = np.nan

    return df


# Process the dataframe
df_beliefs = replace_dropped_task_by_nan(df_beliefs)

df_beliefs.to_excel('output/beliefs.xlsx', index=False)


# Function to print rows with dropped tasks
def print_rows_with_dropped_tasks(df):
    rows_with_task_dropped = []

    for round_number in range(1, 6):
        pre_task_col = f'questions_seq_groupmove.{round_number}.player.pre_task_dropped'
        post_task_col = f'questions_seq_groupmove.{round_number}.player.post_task_dropped'

        if pre_task_col in df.columns:
            pre_task_dropped_rows = df[df[pre_task_col] == 1].index
            for idx in pre_task_dropped_rows:
                rows_with_task_dropped.append((round_number, idx, 'pre_task_dropped'))

        if post_task_col in df.columns:
            post_task_dropped_rows = df[df[post_task_col] == 1].index
            for idx in post_task_dropped_rows:
                rows_with_task_dropped.append((round_number, idx, 'post_task_dropped'))

    # Sort the rows with task dropped based on session, computer number, round number, and task type
    sorted_rows = sorted(rows_with_task_dropped,
                         key=lambda x: (df.loc[x[1], 'session'], df.loc[x[1], 'computer_number'], x[0], x[2]))

    for round_number, index, task_type in sorted_rows:
        row = df.loc[index]
        print(
            f'Session: {row.session}, Computer Number: {row.computer_number}, Round Number: {round_number}, Task Type: {task_type}')


# Example usage with the dataframe df_beliefs
print_rows_with_dropped_tasks(df_beliefs)


#### UNSHUFFLE THE BELIEFS DATA ####

def get_inverse_seq(seq):
    inverse_seq = {}  # Dictionary to store the inverse sequences

    # Extract and inversely sort the sequence for each computer_number
    for key in seq:
        # Original sequence
        original_seq = seq[str(key)]

        # Convert the sequence to a 'normal' sequence starting from 0
        original_seq_zero_based = [i - 1 for i in original_seq]

        # Get inverse sequence with argsort and store it
        inv_seq = list(np.argsort(original_seq_zero_based) + 1)  # '+ 1' to bring sequence back to 1..N range after argsorting
        inverse_seq[key] = inv_seq

    return inverse_seq


# load seq:
with open('input/seq.json', 'r') as f:
    seq = json.load(f)

# Get the inverse sequence
inverse_seq = get_inverse_seq(seq)

# Make a copy of df_beliefs
df_beliefs_copy = df_beliefs.copy()

# Remove columns that do not start with "questions_seq_groupmove."
columns_to_keep = [col for col in df_beliefs_copy.columns if
                   col.startswith("questions_seq_groupmove.") or col == "computer_number" or col == "session"]
df_questions = df_beliefs_copy[columns_to_keep]

# Create an empty dataframe to store the unshuffled data
df_unshuffled = pd.DataFrame()

# Iterate through each computer number in inverse_seq
for computer_number, seq in inverse_seq.items():
    # Filter df_questions for the current computer_number
    df_subset = df_questions[df_questions['computer_number'] == int(computer_number)].copy()

    if not df_subset.empty:
        # Create a dictionary to store the new column names
        new_columns = {}

        # Iterate through the sequence
        for new_round, old_round in enumerate(seq, 1):
            # Find columns for the old round
            old_columns = [col for col in df_subset.columns if col.startswith(f"questions_seq_groupmove.{old_round}.")]

            # Create new column names
            for col in old_columns:
                new_col = col.replace(f".{old_round}.", f".{new_round}.")
                new_columns[col] = new_col

        # Rename the columns for this computer number
        df_subset.rename(columns=new_columns, inplace=True)
        df_unshuffled = pd.concat([df_unshuffled, df_subset])

# Step 5: Combine non-question columns and the new unshuffled dataframe together
non_question_columns = [col for col in df_beliefs_copy.columns
                        if not col.startswith("questions_seq_groupmove.")
                        or col in ['computer_number', 'session']]
df_non_questions = df_beliefs_copy[non_question_columns]

# Ensure 'session' column exists in df_non_questions
if 'session' not in df_non_questions.columns:
    df_non_questions['session'] = df_non_questions['computer_number'].apply(
        lambda x: f"{'O' if int(x) % 2 == 1 else 'X'}_session{x}")

# Merge based on computer_number and session
df_beliefs_unshuffled = pd.merge(df_non_questions, df_unshuffled, on=['computer_number', 'session'], how="left")

# Verify final shape
print("Final df_beliefs_unshuffled shape: ", df_beliefs_unshuffled.shape)

# Additional verification
print(df_beliefs_unshuffled['computer_number'].value_counts())

df_beliefs_unshuffled.to_excel('output/beliefs_unshuffled.xlsx', index=False)

# Additional verification
print(df_beliefs_unshuffled['computer_number'].value_counts())