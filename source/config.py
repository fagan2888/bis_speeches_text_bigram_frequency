from utils import *
import os

# User configuration
sentence_length_outlier = [0, 1, 2, 181, 252]
bigram_window_size = 15
bigram_max_rank = None
stopword_list = ['financial', 'market', 'federal', 'bank', 'banking', 'bankers', 'speech', 'bi', 'review', 'year',
                 'reserve', 'policy', 'state', 'central', 'board', 'percent', 'rate'
    , 'mr', 'alan', 'greenspan', 'ben', 'bernanke', 'janet', 'yellen', 'jerome', 'powell', 'vol'
    , 'ha', 'wa', 'ii']
ws_quarterly, ws_semiannually, ws_annually = 4, 2, 1
strength_alpha = 0.9

# System configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('base_dir:', base_dir)

data_dir = os.path.join(base_dir, 'data')
output_base_dir = os.path.join(base_dir, 'output')
create_dirs([output_base_dir])


class Parameters:
    def __init__(self):
        self.sentence_length_outlier = sentence_length_outlier
        self.bigram_window_size = bigram_window_size
        self.bigram_max_rank = bigram_max_rank
        self.stopword_list = stopword_list
        self.ws_quarterly = ws_quarterly
        self.ws_semiannually = ws_semiannually
        self.ws_annually = ws_annually
        self.strength_alpha = strength_alpha
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.output_base_dir = output_base_dir
        self.bis_raw_pkl_filepath = os.path.join(data_dir, 'bis_w_content_FINAL.pkl')
        self.fred_gdp_quarterly_csv_filepath = os.path.join(data_dir, 'fred_gdp_quarterly.csv')
        self.fred_gdp_semiannually_csv_filepath = os.path.join(data_dir, 'fred_gdp_semiannually.csv')
        self.fred_gdp_annually_csv_filepath = os.path.join(data_dir, 'fred_gdp_annually.csv')
        self.fred_unemp_quarterly_csv_filepath = os.path.join(data_dir, 'fred_unemp_quarterly.csv')

    def __str__(self):
        item_strf = ['{} = {}'.format(attribute, value) for attribute, value in self.__dict__.items()]
        strf = 'Parameters(\n  {}\n)'.format('\n  '.join(item_strf))
        return strf


parameters = Parameters()
print(parameters)
