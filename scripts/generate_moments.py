import sys
import os
import argparse

sys.path.append(os.path.abspath(os.curdir))
sys.path.append(os.path.abspath(os.curdir.join(["src"])))

import cv2
from src.core import ProcessingPipeNetwork
from src.preprocessing import (
    SkinColorSegmenterModule,
    GaussianBlurModule,
    EdgeDetectorModule,
    ThresholdFillModule,
    HoleClosingModule,
    EdgeSmoothingModule, 
    ObjectSeparatorModule,
    DownscaleModule
    )
from src.feature_extraction import HuMomentsExtractor
from src.classification import DummyClassifier

import pandas as pd

def process(inp, outp):
    pipe=ProcessingPipeNetwork()

    pipe.add_preprocessing(DownscaleModule(), 'downscale')
    pipe.add_preprocessing(SkinColorSegmenterModule('models/skin_segmentation/model1'))
    pipe.add_preprocessing(GaussianBlurModule())
    pipe.add_preprocessing(EdgeDetectorModule(lower_thresh=0, upper_thresh=40))
    pipe.add_preprocessing(ThresholdFillModule(), name='thresh', deps=['GaussianBlur', 'EdgeDetector'])
    pipe.add_preprocessing(HoleClosingModule(), name='hole')
    pipe.add_preprocessing(EdgeSmoothingModule(), name='smooth')
    pipe.add_preprocessing(ObjectSeparatorModule(), name='sep')

    pipe.set_feature_extractor(HuMomentsExtractor())
    pipe.set_classifier(DummyClassifier())

    df=[]

    for gesture_type in ['rock', 'paper', 'scissors']:
        print('starting to process', gesture_type)

        folder=os.path.join(inp, gesture_type)
        fail=0

        for path in os.listdir(folder):
            data=cv2.imread(os.path.join(folder, path))
            data=cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

            pipe.process_full_pipeline(data)

            features=pipe.get_feature_vector()

            # a nem pontosan 1 felismert kéz itt biztos hiba, így ezeket a képeket nem tesszük bele az adathalmazba
            if isinstance(features, list) and len(features)==1:
                features=features[0]
            if isinstance(features, list) or features is None:
                fail=fail+1
                continue
            features=features.features

            df.append([features, gesture_type, False])

            if len(df)%100==0:
                print(len(df), 'images processed so far')
        
        print('finished processing', gesture_type, ':', fail, 'failed images')
    
    df=pd.DataFrame(data=df, columns=['features', 'label', 'test'])
    df=df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.loc[round(len(df)*0.95):, 'test']=True

    print('processing done, saving data...')
    df.to_pickle(os.path.join(outp, 'hu_moments.pkl'))
    print('Done!')

def main():
    parser = argparse.ArgumentParser(
        description='Generate Hu moments of train datasets.',
        epilog="""
Example:
  python scripts/generate_moments.py --src data/raw --dst data/processed
        """
    )

    parser.add_argument(
        '--src',
        type=str,
        help='Root folder of the train dataset'
    )

    parser.add_argument(
        '--dst',
        type=str,
        help='Folder to put the results into'
    )

    args = parser.parse_args()
    if not args.src or not args.dst:
        raise Exception('Source and destination folders must be given')
    process(args.src, args.dst)

if __name__ == '__main__':
    main()