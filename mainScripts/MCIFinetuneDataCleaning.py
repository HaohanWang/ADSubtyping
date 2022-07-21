from os.path import join
from collections import defaultdict
import numpy as np

# Data pipeline to prepare subjects used for model finetuning with MCI -> AD progression

READ_DIR = '/Users/gn03249822/Desktop/CMU/DirectedStudy/AlzheimerData'


# return the list of subjects that satisfy our criteria for finetuning
def find_mci_subjects(img_dir=READ_DIR + 'ADNI_CAPS'):
    subjects_to_labels = defaultdict(list)  # subject: [label]
    subjects_to_split = defaultdict(set)


    mci_subjects_to_new_label = {}

    csv_path = join(img_dir, 'split.pretrained.0.csv')
    text = [line.strip() for line in open(csv_path)]

    for line in text[1:]:
        items = line.split(',')

        split = items[-1]
        csv_label = items[-2]
        subject = items[0]

        subjects_to_split[subject].add(split)
        subjects_to_labels[subject].append(csv_label)

        assert len(subjects_to_split[subject]) <= 1  # sanity check to make sure a subject does not appear in more than one split


    for sub, labels in subjects_to_labels.items():
        if "MCI" in labels:
            mci_subjects_to_new_label[sub] = 1 if labels[-1] == 'AD' else 0
    print(f"{len(mci_subjects_to_new_label)} MCI subjects found")

    return mci_subjects_to_new_label


def generate_mci_csv(img_dir=READ_DIR + 'ADNI_CAPS'):
    mci_subjects_to_new_label = find_mci_subjects(img_dir)
    original_csv_path = join(img_dir, 'split.pretrained.0.csv')
    text = [line.strip() for line in open(original_csv_path)]

    with open(READ_DIR + 'ADNI_CAPS/mci_finetune_clean.csv', 'w') as file:
        for line in text[1:]:
            items = line.split(',')

            subject = items[0]
            session = items[1]
            age = items[2]
            gender = items[3]
            original_label = items[4]

            if subject in mci_subjects_to_new_label and original_label == 'MCI':
                # split = np.random.choice(['train', 'val', 'test'], p=[0.8, 0.1, 0.1])

                new_label = 'AD' if mci_subjects_to_new_label[subject] == 1 else 'CN'
                file.writelines(','.join([subject, session, age, gender, new_label]) + '\n')


        for line in text[1:]:





# generate_mci_csv()

