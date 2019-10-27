import pickle 
from datetime import datetime,timedelta
import os
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords 
from nltk import pos_tag

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

alphabet_regex = re.compile('[^a-zA-Z]')
doublespace_pattern = re.compile('\s+')
def get_sentences(content, lower_case=False, alphabet_only=False):
    if lower_case:
        content = content.lower()
            
    sentences = sent_tokenize(content)
    for i in range(len(sentences)):
        if alphabet_only:
            sentences[i] = alphabet_regex.sub(' ', sentences[i])
            sentences[i] = doublespace_pattern.sub(' ', sentences[i])
        sentences[i] = sentences[i].strip()    
    return sentences

noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
adjective_tags = ['JJ', 'JJR', 'JJS']
adverb_tags = ['RB', 'RBR', 'RBS']
verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
def get_pos_tagged_words(content, lower_case=False, lemmatize=False, stemming=False, alphabet_only=False, remove_len_one=False, include_noun=True, include_adjective=True, include_adverb=True, include_verb=True):
    if lower_case:
        content = content.lower()
    words = word_tokenize(content)
    
    lemmatizer = WordNetLemmatizer() 
    porter_stemmer  = PorterStemmer()
    for i in range(len(words)):
        if lemmatize:
            words[i] = lemmatizer.lemmatize(words[i])
        if stemming:
            words[i] = porter_stemmer.stem(words[i])
        if alphabet_only:
            words[i] = alphabet_regex.sub(' ', words[i])
            words[i] = doublespace_pattern.sub(' ', words[i])
        words[i] = words[i].strip()
    words = [word for word in words if word != '' ]
    
    pos_tagged_words = pos_tag(words)
    if remove_len_one:
        pos_tagged_words = [(x,y) for (x,y) in pos_tagged_words if len(x) != 1 ]
    
    tag_list = list()
    if include_noun:
        tag_list.extend(noun_tags)
    if include_adjective:
        tag_list.extend(adjective_tags)
    if include_adverb:
        tag_list.extend(adverb_tags)
    if include_verb:
        tag_list.extend(verb_tags)
    if len(tag_list) != 0:
        pos_tagged_words = [(x,y) for (x,y) in pos_tagged_words if y in tag_list]
    else:
        pos_tagged_words = [(x,y) for (x,y) in pos_tagged_words if y not in ('CC', 'IN', 'DT', 'TO')]
    
    return pos_tagged_words

def _period_dict(target_dict, _period, start, end):
    for _month in [str('%02d' % i) for i in range(start, end+1)]:
        target_dict[_month] = _period
    return target_dict

def quarterly_dict():
    target_dict = dict()
    _period, start, end = 'Q1', 1, 3
    _period_dict(target_dict, _period, start, end)
    _period, start, end = 'Q2', 4, 6
    _period_dict(target_dict, _period, start, end)
    _period, start, end = 'Q3', 7, 9
    _period_dict(target_dict, _period, start, end)
    _period, start, end = 'Q4', 10, 12
    _period_dict(target_dict, _period, start, end)
    return target_dict

def semiannually_dict():
    target_dict = dict()
    _period, start, end = '1H', 1, 6
    _period_dict(target_dict, _period, start, end)
    _period, start, end = '2H', 7, 12
    _period_dict(target_dict, _period, start, end)
    return target_dict

def annually_dict():
    target_dict = dict()
    _period, start, end = '', 1, 12
    _period_dict(target_dict, _period, start, end)
    return target_dict

def dict_val_as_list_append(target_dict, index_key, val):
    if index_key not in target_dict:
        target_dict[index_key] = list()
    target_dict[index_key].append(val)
    return target_dict

def get_grouped_list_in_dict(target_list, period='quarterly'): # 'quarterly', 'semiannually', 'annually'
    if period == 'quarterly':
        period_dict = quarterly_dict()
    elif period == 'semiannually':
        period_dict = semiannually_dict()
    elif period == 'annually':
        period_dict = annually_dict()
    
    result_dict = dict()
    for _item in target_list:
        _date = _item['date']
        _year = _date[:4]
        _month = _date[5:7]
        grouping_key = get_str_concat(_year, period_dict[_month])
        result_dict = dict_val_as_list_append(result_dict, grouping_key, _item)
    return result_dict