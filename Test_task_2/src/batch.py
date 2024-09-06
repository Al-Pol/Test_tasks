import pandas as pd
from model import get_result

def main():
    df = pd.read_excel('../data/DS_NLP.xlsx', index_col=0)
    text = df['text']
    predict = get_result(text)
    predict.to_csv('predictions.csv')


if __name__ == "__main__":
    main()