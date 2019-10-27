import pickle 
from datetime import datetime,timedelta
import os

def create_dirs(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        current_pkl = pickle.load(f)
    print('Completed loading:', filepath)
    return current_pkl

def end_pkl(target_to_save, pkl_path, start=None):
    with open(pkl_path, 'wb') as f:
        pickle.dump(target_to_save, f)
    print('Creating .pkl completed: ', pkl_path)

    if start is not None:
        elapsed_time = time.time() - start
        elapsed_time_format = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print('END. Elapsed time: ', elapsed_time_format)

def get_now_time_str():
    return datetime.now().strftime("%Y%m%d-%H-%M-%S")

def get_str_concat(*args):
    _str = ""
    firstLine = True
    for idx, arg in enumerate(args):
        if idx == 0:
            _str += arg
            firstLine = False
            continue
        _str = _str + "_" + arg
    return _str        
        
def dates_between(start_dt, end_dt):
    dates = []
    cursor = start_dt
    while cursor <= end_dt:
        if cursor.strftime('%Y-%m-%d') not in dates:
            dates.append(cursor.strftime('%Y-%m-%d'))
        cursor += timedelta(days=1)
    return dates
    
def get_target_list_filtered_by_author_and_years(content_dict, target_author, target_dates):
    target_list = list()
    for _key in content_dict.keys():
        _date = content_dict[_key]['date']
        _title = content_dict[_key]['short_info']

        if _date in target_dates and target_author in _title:
            target_list.append(content_dict[_key])
    return target_list