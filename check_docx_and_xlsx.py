import os


def check_missing_xls_files(session_name=None):
    data_dir = 'input/X/chatlog'
    sessions = [dir_name for dir_name in os.listdir(data_dir) if 'X_session' in dir_name]

    if session_name:
        if session_name in sessions:
            sessions = [session_name]
        else:
            return "The specified session does not exist."

    missing_files = {}

    for session in sessions:
        docx_files = [file_name for file_name in os.listdir(os.path.join(data_dir, session)) if '.docx' in file_name]
        xls_files = [file_name for file_name in os.listdir(os.path.join(data_dir, session)) if '.xlsx' in file_name]

        for docx_file in docx_files:
            corresponding_xls_file = docx_file.replace('.docx', '.xlsx')
            if corresponding_xls_file not in xls_files:
                if session in missing_files:
                    missing_files[session].append(corresponding_xls_file)
                else:
                    missing_files[session] = [corresponding_xls_file]

    return missing_files


missing_files = check_missing_xls_files()
print(missing_files)
