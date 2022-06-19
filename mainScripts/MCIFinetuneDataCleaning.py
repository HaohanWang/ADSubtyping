from os.path import join
from collections import defaultdict

# Data pipeline to prepare subjects used for model finetuning with MCI -> AD progression

READ_DIR = '/home/ec2-user/mnt/home/ec2-user/alzstudy/AlzheimerData/'


# return the list of subjects that satisfy our criteria for finetuning
def find_mci_subjects(img_dir=READ_DIR + 'ADNI_CAPS', idx_fold=0):
    subjects_to_labels = defaultdict(list)  # subject: [label]
    subjects_to_split = defaultdict(set)
    mci_subjects_to_new_label = {}

    csv_path = join(img_dir, f'split.pretrained.{idx_fold}.csv')
    text = [line.strip() for line in open(csv_path)]

    for line in text[1:]:
        items = line.split(',')

        split = items[-1]
        csv_label = items[-2]
        subject = items[0]

        subjects_to_split[subject].add(split)
        subjects_to_labels[subject].append(csv_label)

        assert len(subjects_to_split[subject]) <= 1  # sanity check to make sure a subject does not
        # appear in more than one split

    # print(subjects_to_labels)

    for sub, labels in subjects_to_labels.items():
        if "MCI" in labels:
            mci_subjects_to_new_label[sub] = 1 if labels[-1] == 'AD' else 0

    print(f"{len(mci_subjects_to_new_label)} MCI subjects found")
    return mci_subjects_to_new_label



