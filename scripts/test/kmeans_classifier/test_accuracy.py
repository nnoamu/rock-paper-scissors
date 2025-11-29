import sys
import os
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(os.curdir))
sys.path.append(os.path.abspath(os.curdir.join(["src"])))

from src.core import FeatureVector, FeatureType
from src.classification import KMeansClassifier
from src.core import ClassificationResult
from typing import List
import pandas as pd
import numpy as np

M={'rock': 0, 'paper': 1, 'scissors': 2}

def evaluate(k, inp):
    classifier=KMeansClassifier(k, inp, test=True)
    conf_mtx=np.zeros((3, 3), dtype=np.int32)

    df=pd.DataFrame(columns=['features', 'label', 'test'])
    for pth in inp:
        akt=pd.read_pickle(pth)
        akt=akt[akt['test']]
        df=pd.concat([df, akt], ignore_index=True, sort=False)
    
    for sample in df.itertuples(index=False):
        result=classifier.classify(FeatureVector(FeatureType.GEOMETRIC, 'none', 0, np.array(sample.features)))

        if isinstance(result, ClassificationResult):
            conf_mtx[M[sample.label], M[result.predicted_class.value]]=conf_mtx[M[sample.label], M[result.predicted_class.value]]+1
        else: #list
            mx_label='rock'
            mx_val=-1
            for x in result:
                if x.confidence>mx_val:
                    mx_val=x.confidence
                    mx_label=x.predicted_class.value
            conf_mtx[M[sample.label], M[mx_label]]=conf_mtx[M[sample.label], M[mx_label]]+1
    
    print('accuracy:', str((conf_mtx[0, 0]+conf_mtx[1, 1]+conf_mtx[2, 2])/len(df)*100)+'%')
    print('confusion matrix:')
    print(conf_mtx)

def main():
    parser = argparse.ArgumentParser(
        description='Generate Hu moments of train datasets.',
        epilog="""
Example:
  python scripts/test/kmeans_classifier/test_accuracy.py --k 5 --data data/processed/drgfreeman/hu_moments.pkl data/processed/hectorandac/hu_moments.pkl
        """
    )

    parser.add_argument(
        '--k',
        type=int,
        help='Number of closest vectors to consider.'
    )

    parser.add_argument(
        '--data',
        type=str,
        nargs='+',
        help='.pkl file containing the labeled feature vectors'
    )

    args = parser.parse_args()
    if not args.k:
        raise Exception('The value k must be given')
    if not args.data:
        raise Exception('At least 1 .pkl data file must be given')
    evaluate(args.k, args.data)

if __name__ == '__main__':
    main()