from config import parameters
from utils import *
import os
import re

base_dir = parameters.base_dir
data_dir = parameters.data_dir
output_base_dir = parameters.output_base_dir
bis_raw_pkl_filepath = parameters.bis_raw_pkl_filepath

temp_dir = os.path.join(output_base_dir, 'temp')
os.system('rm -rf ' + temp_dir)
create_dirs([temp_dir])
target_list_filepath = os.path.join(temp_dir, get_str_concat('target-list', get_now_time_str()) + '.pkl')
target_list_units_filepath = os.path.join(output_base_dir, get_str_concat('target-list-units', get_now_time_str()) + '.pkl')

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
    end_pkl(target_list, target_list_filepath)
    
    # Noun, Adjective, Adverb, Verb only
    count = 0
    for one_doc in target_list:   
        count += 1
        print('[',count,'/',len(target_list),'] Processing', one_doc['key'])
        
        _content = re.sub("\d+", " ", one_doc['content'])   # remove numbers
        one_doc['sentences'] = get_sentences(_content, lower_case=False, alphabet_only=False)
        
        one_doc['unigrams_by_sentence'] = list()
        for sentence in one_doc['sentences']:
            pos_tagged_words = get_pos_tagged_words(sentence, lower_case=True, lemmatize=True, stemming=False, alphabet_only=True, remove_len_one=True, include_noun=True, include_adjective=True, include_adverb=True, include_verb=True)
            one_doc['unigrams_by_sentence'].append(pos_tagged_words)

    os.system('rm -rf ' + temp_dir)
    end_pkl(target_list, target_list_units_filepath)
    
    # grouping: annualy, semi-annualy, quarterly
    for _period in ['quarterly', 'semiannually', 'annually']:
        target_quarterly_dict = get_grouped_list_in_dict(target_list, period=_period)
        end_pkl(target_quarterly_dict, os.path.join(output_base_dir, get_str_concat('target-list-units-grouped', _period, get_now_time_str()) + '.pkl'))
    
if __name__ == '__main__':
    main()