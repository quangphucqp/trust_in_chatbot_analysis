import os
import pandas as pd
import numpy as np
import json


# Methods

def load_files(session_name=None):

    # Load chatlog data from the directory
    # By default, load all sessions
    # If `session_name` is specified, only load the data from that session

    data_dir = 'input/O/rec'
    sessions = [dir_name for dir_name in os.listdir(data_dir) if 'O_session' in dir_name]

    if session_name:
        if session_name in sessions:
            sessions = [session_name]
        else:
            return "The specified session does not exist."

    session_data = {}

    for session in sessions:
        file_names = [file_name for file_name in os.listdir(os.path.join(data_dir, session)) if 'xls' in file_name]

        for file_name in file_names:
            file_path = os.path.join(data_dir, session, file_name)

            computer_number = int(''.join(filter(str.isdigit, file_name.split('_')[1])))
            df = pd.read_excel(file_path)

            session_data[computer_number] = df

    return session_data


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


def unshuffle_rec_and_pivot(chatlog_data, seq):
    inverse_seq = get_inverse_seq(seq)

    unshuffled_chatlog_data = {}  # Dictionary to store the unshuffled chatlog data.

    for computer_number in chatlog_data.keys():
        chatlog = chatlog_data[computer_number]

        if str(computer_number) in inverse_seq:
            # Adjust to 0-based index for DataFrame
            shuffled_indices = [i - 1 for i in inverse_seq[str(computer_number)]]

            # Debugging: Print the computer number and its associated indices
            print(f"Computer number: {computer_number}")
            print(f"Shuffled indices (0-based): {shuffled_indices}")
            print(f"DataFrame length: {len(chatlog)}")
            print(f"Chatlog DataFrame:\n{chatlog}")

            # Check if indices are within bounds
            if max(shuffled_indices) >= len(chatlog):
                print(f"Indices out of bounds for computer number {computer_number}: {shuffled_indices}")
                continue

            # Shuffle the rows:
            try:
                unshuffled_chatlog = chatlog.iloc[shuffled_indices].reset_index(drop=True)
                print(f"Unshuffled chatlog:\n{unshuffled_chatlog}")
            except Exception as e:
                print(f"Error while unshuffling data for computer number {computer_number}: {e}")
                continue

            # Add index and convert to wide:
            unshuffled_chatlog['question_number'] = unshuffled_chatlog.index + 1

            # Set a temporary index to pivot correctly
            unshuffled_chatlog['tmp_index'] = 0

            unshuffled_chatlog = unshuffled_chatlog.pivot(index='tmp_index', columns='question_number')
            unshuffled_chatlog.columns = [f"{col[0]}.{col[1]}" for col in unshuffled_chatlog.columns]
            unshuffled_chatlog = unshuffled_chatlog.reset_index(drop=True)

            unshuffled_chatlog_data[computer_number] = unshuffled_chatlog  # Update the dictionary with the unshuffled DataFrame
        else:
            print(f"No unshuffled sequence found for computer number {computer_number}. Skipping...")

    return unshuffled_chatlog_data


def pivot_rec_data(chatlog_data_dict):
    processed_chatlog_data = {}  # Dictionary to store the processed chatlog data.

    for computer_id in chatlog_data_dict.keys():
        chatlog = chatlog_data_dict[computer_id]

        # Take the first 5 rows and then reset the index
        processed_chatlog = chatlog.iloc[:5].reset_index(drop=True)

        # Add index and convert to wide:
        processed_chatlog['question_number'] = processed_chatlog.index + 1

        # Set a temporary index to pivot correctly
        processed_chatlog['tmp_index'] = 0

        processed_chatlog = processed_chatlog.pivot(index='tmp_index', columns='question_number')
        processed_chatlog.columns = [f"{col[0]}.{col[1]}" for col in processed_chatlog.columns]
        processed_chatlog = processed_chatlog.reset_index(drop=True)

        processed_chatlog_data[computer_id] = processed_chatlog  # Update the processed DataFrame Dictionary

    return processed_chatlog_data



# load seq:
with open('input/seq.json', 'r') as f:
    seq = json.load(f)

# List to store dataframes for each session
all_dataframes = []

# Loop over session numbers
for session_number in ['O_session1', 'O_session2', 'O_session3', 'O_session4',
                       'O_session5', 'O_session6', 'O_session7', 'O_session8']:

    # Load chatlog data
    chatlog_data = load_files(session_name=session_number)

    # Unshuffle and pivot the chatlog data OR Simply pivot the chatlog data
    chatlog_unshuffled = pivot_rec_data(chatlog_data)

    # List to store dataframes for each session
    all_dataframes = []
    all_unshuffled_dataframes = []

    # Loop over session numbers
    for session_number in ['O_session1', 'O_session2', 'O_session3', 'O_session4',
                           'O_session5', 'O_session6', 'O_session7', 'O_session8']:

        # Load chatlog data
        chatlog_data = load_files(session_name=session_number)

        # Unshuffle and pivot the chatlog data OR Simply pivot the chatlog data
        chatlog_unshuffled = unshuffle_rec_and_pivot(chatlog_data, seq)
        chatlog = pivot_rec_data(chatlog_data)

        # Combine into a single dataframe
        df_list = []
        df_unshuffled_list = []

        for key, value in chatlog.items():
            df = value.reset_index()
            df['computer_number'] = key
            df_list.append(df)

        for key, value in chatlog_unshuffled.items():
            df_unshuffled = value.reset_index()
            df_unshuffled['computer_number'] = key
            df_unshuffled_list.append(df_unshuffled)

        session_df = pd.concat(df_list, ignore_index=True).drop(columns='index')
        session_df['session'] = session_number

        session_unshuffled_df = pd.concat(df_unshuffled_list, ignore_index=True).drop(columns='index')
        session_unshuffled_df['session'] = session_number

        # Append the data frame for this session to the list
        all_dataframes.append(session_df)
        all_unshuffled_dataframes.append(session_unshuffled_df)

    # Create final dataframe by concatenating all session dataframes
    final_df = pd.concat(all_dataframes, ignore_index=True)
    final_unshuffled_df = pd.concat(all_unshuffled_dataframes, ignore_index=True)

    # Save the final dataframe to an Excel file
    final_df.to_excel('output/O_rec.xlsx', index=False)
    final_unshuffled_df.to_excel('output/O_rec_unshuffled.xlsx', index=False)