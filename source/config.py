from utils import *
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('base_dir:', base_dir)

data_dir = os.path.join(base_dir, 'data')
output_base_dir = os.path.join(base_dir, 'output')
create_dirs([output_base_dir])


class Parameters:
    def __init__(self):
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.output_base_dir = output_base_dir
        self.bis_raw_pkl_filepath = os.path.join(data_dir, 'bis_w_content_FINAL.pkl')

    def __str__(self):
        item_strf = ['{} = {}'.format(attribute, value) for attribute, value in self.__dict__.items()]
        strf = 'Parameters(\n  {}\n)'.format('\n  '.join(item_strf))
        return strf


parameters = Parameters()
print(parameters)
