import pandas as pd 
from os.path import join

def wrong(save_dir, split):
    csv_path = join(save_dir,split+'_prediction_info.csv')
    df = pd.read_csv(csv_path)
    df['pred'] = (df['prob_AD'] > 0.5).values.astype(int)
    mispred = df[df['pred'] != df['diagnosis']]
    mispred.to_csv(join(save_dir,'wrong_'+split+'_prediction_info.csv'), index=None)


if __name__ == "__main__":
    save_dir = "prediction"
    splits = ['train', 'val', 'test']
    for split in splits:
        wrong(save_dir, split)