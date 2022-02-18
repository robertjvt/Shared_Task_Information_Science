import numpy as np
import pandas as pd
import scipy.stats


def main():

    df = pd.read_csv('BERT_REG_50_3E-05_20_5_16_MSE_ADAM_TRAIN_REG_DEV_REG_own_testset.csv')
    x = df['Test']
    y = df['Predict']
    rho = scipy.stats.spearmanr(x, y)[0]
    print(rho)


if __name__ == "__main__":
    main()
