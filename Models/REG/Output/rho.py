'''This script will obtain a spearman correlation between the predicted scores and the gold scores'''
import numpy as np
import pandas as pd
import scipy.stats


def main():

    df = pd.read_csv('BERT_100_1E-05_10_3_8_MSE_ADAM_REG.csv')
    x = df['Test']
    y = df['Predict']
    rho = scipy.stats.spearmanr(x, y)[0]
    print(rho)


if __name__ == "__main__":
    main()
