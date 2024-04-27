from tsfresh import extract_features
import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("test_1_converted.csv")
    df = df.fillna(0)
    
    df = df[['id', 'gas', 'gasPrice', 'gasUsed', 'isError', 'timeStamp', 'value']]
    
    extracted = extract_features(df, column_id="id", column_sort="timeStamp")

    # 此處的參數分別是
    # timeseries: data (pandas的DataFrame datatype)
    # column_id : data如何做分類,此處用id欄位做分類
    # column_sort : data如何去做排序
    
    print(extracted)

    return


if __name__ == "__main__":
    main()