from config import parameters
from utils import *
import os

base_dir = parameters.base_dir
data_dir = parameters.data_dir
output_base_dir = parameters.output_base_dir
bis_raw_pkl_filepath = parameters.bis_raw_pkl_filepath

temp_dir = os.path.join(output_base_dir, 'temp')
create_dirs([temp_dir])

def _get_target_dict():
    target_dict = dict()
    target_dict['Greenspan'] = dates_between(datetime.strptime('1987-08-11', '%Y-%m-%d'), datetime.strptime('2006-01-31', '%Y-%m-%d'))
    target_dict['Bernanke'] = dates_between(datetime.strptime('2006-02-01', '%Y-%m-%d'), datetime.strptime('2014-01-31', '%Y-%m-%d'))
    target_dict['Yellen'] = dates_between(datetime.strptime('2014-02-03', '%Y-%m-%d'), datetime.strptime('2018-02-03', '%Y-%m-%d'))
    target_dict['Powell'] = dates_between(datetime.strptime('2018-02-05', '%Y-%m-%d'), datetime.strptime('2019-10-01', '%Y-%m-%d'))
    return target_dict


def main():
    data_dict = load_pkl(bis_raw_pkl_filepath)
    target_dict = _get_target_dict()

    # Target document list
    target_list = list()
    for target_author, target_dates in target_dict.items():
        target_list.extend(get_target_list_filtered_by_author_and_years(data_dict, target_author, target_dates))
    end_pkl(target_list, os.path.join(temp_dir, get_str_concat('target-doc-list', get_now_time_str()) + '.pkl'))
            
if __name__ == '__main__':
    main()