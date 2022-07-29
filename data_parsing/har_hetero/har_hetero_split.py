import pandas as pd
from tqdm import tqdm


def main():
    for patient in tqdm('abcdefghi'):
        process_patient(patient)


def process_patient(patient):
    phone_acc_df = pd.read_csv('data/raw/har_hetero/Phones_accelerometer.csv')
    phone_acc_df = phone_acc_df[phone_acc_df['User'] == patient].reset_index(drop=True)
    phone_gyr_df = pd.read_csv('data/raw/har_hetero/Phones_gyroscope.csv')
    phone_gyr_df = phone_gyr_df[phone_gyr_df['User'] == patient].reset_index(drop=True)
    watch_acc_df = pd.read_csv('data/raw/har_hetero/Watch_accelerometer.csv')
    watch_acc_df = watch_acc_df[watch_acc_df['User'] == patient].reset_index(drop=True)
    watch_gyr_df = pd.read_csv('data/raw/har_hetero/Watch_gyroscope.csv')
    watch_gyr_df = watch_gyr_df[watch_gyr_df['User'] == patient].reset_index(drop=True)

    phone_acc_df.to_csv(f'data/raw/har_hetero/patient_splits/phone_acc_{patient}.csv', index=False)
    phone_gyr_df.to_csv(f'data/raw/har_hetero/patient_splits/phone_gyr_{patient}.csv', index=False)
    watch_acc_df.to_csv(f'data/raw/har_hetero/patient_splits/watch_acc_{patient}.csv', index=False)
    watch_gyr_df.to_csv(f'data/raw/har_hetero/patient_splits/watch_gyr_{patient}.csv', index=False)


if __name__ == '__main__':
    main()
