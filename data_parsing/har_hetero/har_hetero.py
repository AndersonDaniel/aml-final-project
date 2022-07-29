import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.interpolate import interp1d


LABELS = {
    'null': 0,
    'stand': 1,
    'sit': 2,
    'walk': 3,
    'bike': 4,
    'stairsup': 5,
    'stairsdown': 6
}

SAMPLING_RATE_HZ = 100
PHONE_ACC_DEVICES = ['nexus4_1', 'nexus4_2', 's3_1', 's3_2', 's3mini_1', 's3mini_2', 'samsungold_1', 'samsungold_2']
PHONE_GYR_DEVICES = ['nexus4_1', 'nexus4_2', 's3_1', 's3_2', 's3mini_1', 's3mini_2']
WATCH_DEVICES = ['gear_1', 'gear_2', 'lgwatch_1', 'lgwatch_2']


def main():
    for patient in 'abcdefghi':
        parse_patient(patient)


def parse_patient(patient):
    phone_acc_df, phone_gyr_df, watch_acc_df, watch_gyr_df = load_data(patient)

    all_sensor_dfs = [*split_devices(phone_acc_df, PHONE_ACC_DEVICES), *split_devices(phone_gyr_df, PHONE_GYR_DEVICES),
                      *split_devices(watch_acc_df, WATCH_DEVICES), *split_devices(watch_gyr_df, WATCH_DEVICES)]

    all_sensor_dfs = [
        split_segments(add_labels(df))
        for df in all_sensor_dfs
    ]

    res, labels = merge_sensor_dfs(all_sensor_dfs)
    np.save(f'data/parsed/har_hetero/{patient}',
            {'time_series': res, 'labels': labels}, allow_pickle=True)


def split_devices(df, devices):
    return [df[df['Device'] == device] for device in devices]


def add_labels(df):
    if df.shape[0] == 0:
        return df

    return df.assign(label=df['gt'].apply(LABELS.get))


def split_segments(df):
    if df.shape[0] == 0:
        return []

    df['relative_time_seconds'] = (df['Creation_Time'] - df['Creation_Time'].values[0]) / 1e9
    df = df[df['relative_time_seconds'] >= 0].reset_index(drop=True)
    all_segment_indices = np.split(np.arange(df.shape[0]), np.where(df['label'].diff().fillna(1) != 0)[0])[1:]
    res = [
        df.iloc[segment_indices]
        for segment_indices in all_segment_indices
    ]

    res = [
        df for df in res
        if df['relative_time_seconds'].max() - df['relative_time_seconds'].min() >= 10
    ]

    return res


def load_data(patient):
    return map(fill_gt_na, (
        pd.read_csv(f'data/raw/har_hetero/patient_splits/phone_acc_{patient}.csv'),
        pd.read_csv(f'data/raw/har_hetero/patient_splits/phone_gyr_{patient}.csv'),
        pd.read_csv(f'data/raw/har_hetero/patient_splits/watch_acc_{patient}.csv'),
        pd.read_csv(f'data/raw/har_hetero/patient_splits/watch_gyr_{patient}.csv'),
    ))


def fill_gt_na(df):
    df['gt'] = df['gt'].fillna('null')
    return df


def merge_sensor_dfs(all_sensor_dfs):
    res = []
    labels = []
    while True:
        curr_mode = mode([x[0]['label'].values[0] if len(x) > 0 else -1 for x in all_sensor_dfs]).mode[0]
        candidates = [sensor_dfs for sensor_dfs in all_sensor_dfs
                      if len(sensor_dfs) > 0 and sensor_dfs[0]['label'].values[0] == curr_mode]
        if len(candidates) == 0:
            break

        if curr_mode == 0:  # Removing NULL entries
            for sensor_dfs in candidates:
                sensor_dfs.pop(0)

            continue

        start_time, end_time = np.median([x[0]['relative_time_seconds'].values[[0, -1]] for x in candidates], axis=0)
        t = np.linspace(round(start_time), round(end_time),
                        (round(end_time) - round(start_time)) * SAMPLING_RATE_HZ + 1)
        curr_res = []
        empty = True
        for sensor_dfs in all_sensor_dfs:
            signal = np.zeros((t.shape[0], 3))
            if (len(sensor_dfs) > 0 and sensor_dfs[0]['label'].values[0] == curr_mode and
                    iou(start_time, end_time, *sensor_dfs[0]['relative_time_seconds'].values[[0, -1]].tolist()) >= .75):
                df = sensor_dfs.pop(0)
                signal = interp1d(df['relative_time_seconds'], df[['x', 'y', 'z']],
                                  axis=0, kind='linear', bounds_error=False)(t)
                signal = np.nan_to_num(signal, 0)
                empty = False
            else:
                if len(sensor_dfs) > 0 and start_time > sensor_dfs[0]['relative_time_seconds'].max():
                    sensor_dfs.pop(0)

            curr_res.append(signal)

        if not empty:
            res.append(np.concatenate(curr_res, axis=1))
            labels.append(curr_mode)
        else:
            break

    return res, labels


def iou(x1, x2, y1, y2):
    union_range = max(x2, y2) - min(x1, y1)
    intersection_range = min(x2, y2) - max(x1, y1)
    return intersection_range / union_range


if __name__ == '__main__':
    main()
