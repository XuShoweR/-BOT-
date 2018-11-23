import os
import time
import numpy as np
import json
import argparse
import logging

# Log config
log_file=os.path.join('./',str(time.strftime('%Y-%m-%d-%H_%M_%S',time.localtime()))+'.log')
log_format='%(asctime)s - %(message)s'
logger=logging.getLogger('winner-bot-eval')

logging.basicConfig(level=logging.DEBUG)
hnd_logger=logging.FileHandler(log_file)
hnd_logger.setLevel(level=logging.DEBUG)
logger_formatter=logging.Formatter(log_format)
hnd_logger.setFormatter(logger_formatter)
logger.addHandler(hnd_logger)

class Eval(object):
    def __init__(self):
        print('start')

    def _check_key_is_valid(self, d):
        # check if an object's label is valid
        is_valid = False
        for i, x in enumerate(['minx', 'maxx', 'maxy', 'male', 'female',
                  'staff', 'customer', 'stand', 'sit', 'play_with_phone']):
            if x not in d.keys():
                logger.info('ERROR: Predict file\'s format is not correct. Atrribute name is not correct.')
                return is_valid
            if not isinstance(d[x], int) and not isinstance(d[x], float):
                logger.info('ERROR: Predict file\'s format is not correct. Value type must be int or float.')
                return is_valid
        is_valid = True
        return is_valid

    def _check_file_format(self,json_data):
        # check label file format 
        try:
            if 'results' in json_data.keys():
                for label in json_data['results']:
                    for i, obj in enumerate(label['object']):
                        if self._check_key_is_valid(obj) is False:
                            logger.info('ERROR: Predict file\'s format is not correct. Something wrong in ''object''.')
                            return False            
            else:
                logger.info('ERROR: Predict file\'s format is not correct. No key name ''results'' has found.')
                return False
        except:
            logger.info('ERROR: Predict file\'s format is not correct.')
            return False
        return True

    def eval(self,predict_file_path):
        # evaluation
        with open(predict_file_path) as pred_file:
            try:
                pred_data=json.load(pred_file)
            except:
                logger.info('ERROR: Predict file\'s format is not correct. It is not a json file')
                return 0,0

            # check format
            if self._check_file_format(pred_data) is False:
                logger.info('File format error.')
                return 0,0
            logger.info('File format checking pass.')

    
def runApp(args):
    # evaluation 
    ff=Eval()
    ff.eval(args.f)
    

if __name__=='__main__':
    # parse argument
    parser=argparse.ArgumentParser()
    parser.add_argument('--f',default='Fight_val2_20181010.json',type=str)
    args=parser.parse_args()
    
    # default path is working path
    predict_file_path=os.path.join(os.path.dirname(__file__),args.f)

    logger.info('Evaluate file : %s'%(predict_file_path))
    if os.path.exists(predict_file_path) is False:
        logger.info('ERROR: Input file %s is not exist.'%(predict_file_path))
    else:
        runApp(args)




