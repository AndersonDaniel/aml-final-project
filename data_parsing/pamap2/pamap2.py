import numpy as np
import pandas as pd


def process_subject(subject_no):
    df = pd.read_csv(f'data/raw/pamap2/Protocol/subject{subject_no}.dat', header=None, delimiter=' ')
    segments = split_signals(df)
    labels = []
    res = []
    for segment in segments:
        curr_res, curr_label = sample(segment)
        res.append(curr_res)
        labels.append(curr_label)

    np.save(f'data/parsed/pamap2/{subject_no}',
            {'time_series': res, 'labels': labels}, allow_pickle=True)


def split_signals(df):
    labels = df[df.columns[1]]
    return [df.iloc[idx] for idx in np.split(np.arange(df.shape[0]), np.where(labels != labels.shift(1))[0][1:])]


def sample(df):
    label = df[df.columns[1]].values[0]
    signal_columns = [df.columns[2]] + sum([
        df.columns[i:i+4].tolist() + df.columns[i+7:i+13].tolist()
        for i in [3, 20, 37]
    ], [])

    return df[signal_columns].fillna(0).values, label


def main():
    for i in range(9):
        process_subject(i + 101)


if __name__ == '__main__':
    main()
