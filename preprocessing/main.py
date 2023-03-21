import os
from audioset_preprocessor import AS_processor
from constants import DATASET

def main():
    AS_processor(os.path.join(DATASET, 'audioset'))

if __name__ == '__main__':
    main()