"""
K-közép osztályozó euklideszi távolság alapján. Egyenlőség esetén az osztályonként átlag távolság dönt.
"""

import time
import numpy as np
import pandas as pd
from typing import List
from core import BaseClassifier, FeatureVector, ClassificationResult, GestureClass

class KMeansClassifier(BaseClassifier):

    def __init__(self, k: int, data_path: str | List[str], test: bool=False):
        super().__init__(name="KMeans")
        self.k=k
        self.df=None

        if isinstance(data_path, str):
            self.df=pd.read_pickle(data_path)
            if test:
                self.df=self.df[self.df['test']==False]
        else:
            for pth in data_path:
                df=pd.read_pickle(pth)
                if test:
                    df=df[df['test']==False]
                if self.df is None:
                    self.df=df
                else:
                    self.df=pd.concat([self.df, df], ignore_index=True, sort=False)
                

    def _process(self, input: FeatureVector) -> ClassificationResult:
        start=time.perf_counter()

        data=input.features
        best=[]
        for sample in self.df.itertuples(index=False):
            features=np.array(sample.features)
            features=np.abs(data-features)
            distance=np.sqrt(np.sum(features*features))
            
            best.append((distance, sample.label))
            idx=len(best)-1
            while idx>0 and best[idx][0]<best[idx-1][0]:
                best[idx], best[idx-1]=best[idx-1], best[idx]
                idx=idx-1
            if len(best)>self.k:
                best.pop()
        
        counts={'rock': 0, 'paper': 0, 'scissors': 0}
        sum_dist={'rock': 0, 'paper': 0, 'scissors': 0}
        for distance, label in best:
            counts[label]=counts[label]+1
            sum_dist[label]=sum_dist[label]+distance
        
        mx_count=-1
        mx_label='rock'
        single_mx=False
        for label, cnt in counts.items():
            if cnt>mx_count:
                mx_count=cnt
                mx_label=label
                single_mx=True
            elif cnt==mx_count:
                single_mx=False
        
        mn_avg=1e10
        if not single_mx:
            for label, cnt in counts.items():
                avg=sum_dist[label]/cnt
                if cnt==mx_count and mn_avg>avg:
                    mx_label=label
                    mn_avg=avg

        gesture=GestureClass.ROCK if mx_label=='rock' else (GestureClass.PAPER if mx_label=='paper' else GestureClass.SCISSORS)
        result=ClassificationResult(
            gesture, mx_count/self.k,
            {
                GestureClass.ROCK: counts['rock']/self.k,
                GestureClass.PAPER: counts['paper']/self.k,
                GestureClass.SCISSORS: counts['scissors']/self.k,
                GestureClass.UNKNOWN: 0
            },
            -1,
            self.name
            )
        
        process_time=(time.perf_counter()-start)*1000
        result.processing_time_ms=process_time
        return result