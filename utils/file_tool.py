import os
import json
import shutil
import pickle
import pandas as pd


def get_path_filename(path, suffix='.txt'):
    file_path = path.split("/")[-1].replace(suffix, '')
    return file_path


def copy_file(source_path, des_path):
    shutil.copy2(source_path, des_path)


def foreach_file(m_dir, prefix):
    dirs = []
    path_dir = os.listdir(m_dir)
    for file_name in path_dir:
        if prefix in file_name:
            child = os.path.join('%s%s' % (m_dir, file_name))
            dirs.append(child)
    return dirs


def recursive_foreach_file(m_dir, prefix):
    file_list = []
    for root, dirs, files in os.walk(m_dir):
        for file in files:
            if prefix in file:
                file_list.append(os.path.join(root, file))
    return file_list


def recursive_delete_file(m_dir, prefix):
    for root, dirs, files in os.walk(m_dir):
        for file in files:
            if prefix in file:
                os.remove(os.path.join(root, file))


def dict_save_to_json(data_dict, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)


def json_read_to_dict(path):
    with open(path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    return data_dict


def save_to_plk(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_from_plk(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def dict_save_to_excel(data_dict, file_path):
    df = pd.DataFrame(data_dict)
    df.to_excel(file_path, index=False, engine='openpyxl')


def dict_read_from_excel(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    data_dict = df.to_dict('list')
    return data_dict


def dict_from_xlsx_col(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    column_dict = {}
    for column in df.columns:
        column_dict[column] = df[column].tolist()
    return column_dict
