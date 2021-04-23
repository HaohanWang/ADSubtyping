import os
from os.path import join, basename, exists
from os import makedirs
import errno
from multiprocessing.pool import Pool
from tqdm import tqdm
import argparse
import numpy as np


def crop_nifti(input_img, ref_crop, output_path):

    import nibabel as nib
    import numpy as np
    from nilearn.image import resample_img, resample_to_img
    from nibabel.spatialimages import SpatialImage

    # basedir = os.getcwd()
    ## resample the individual MRI onto the cropped template image
    crop_img = resample_to_img(input_img, ref_crop)
    crop_img.to_filename(output_path)


def get_sub_id(t1w_path):
    import re
    m1 = re.search(r'([0-9]+_S_[0-9]+)', t1w_path)
    m2 = re.search(r'(200[0-9]+-[0-9]+-[0-9]+)', t1w_path)
    if m1 is None or m2 is None:
        raise ValueError('Input filename: {} can not find info.'.format(t1w_path))
    subject = m1.group(1)
    session = m2.group(1)
    return subject, session

def preprocess_single_t1w(img_path, out_dir, ref_template, ref_crop):

    from nipype.interfaces.ants import N4BiasFieldCorrection, RegistrationSynQuick
    import tempfile

    try:
        tmp_dir = tempfile.mkdtemp(dir='tmp')
        subject, session = get_sub_id(img_path)
        outpath = join(out_dir, subject, session, basename(img_path).split('.nii')[0]+'_n4Corrected_affineWarped_Cropped.nii.gz')
        if exists(outpath):
            print('File: {} exists!'.format(outpath))
            return True        

        n4biascorrection = N4BiasFieldCorrection()
        n4biascorrection.inputs.dimension = 3
        n4biascorrection.inputs.bspline_fitting_distance=600
        n4biascorrection.inputs.input_image = img_path
        n4biascorrection_output = join(tmp_dir, basename(img_path).split('.nii')[0]+'_n4corrected.nii')
        print(n4biascorrection_output)
        n4biascorrection.inputs.output_image = n4biascorrection_output
        os.system(n4biascorrection.cmdline+' > /dev/null')

        ants_registration = RegistrationSynQuick()
        ants_registration.inputs.dimension = 3
        ants_registration.inputs.transform_type = 'a'
        ants_registration.inputs.fixed_image = ref_template
        ants_registration.inputs.moving_image = n4biascorrection_output
        ants_registration.inputs.output_prefix = join(tmp_dir, basename(img_path).split('.nii')[0]+'_n4corrected_affine')
        ants_output = join(tmp_dir, basename(img_path).split('.nii')[0]+'_n4corrected_affineWarped.nii.gz')

        makedirs(join(out_dir, subject, session), exist_ok=True)
        # outpath = join(out_dir, subject, sessioni, basename(img_path).split('.nii')[0]+'_n4Corrected_affineWarped_Cropped.nii.gz')
        # print(n4biascorrection.cmdline)
        os.system(ants_registration.cmdline+' > /dev/null')
        crop_nifti(ants_output, ref_crop, outpath)

        os.system('rm -r ' + tmp_dir)

        return True

    except Exception as e:

        print(e)
        return False



def preprocess_t1w(pid, img_path_list, out_dir, ref_template, ref_crop):

    failed_id = []
    for img_path in tqdm(img_path_list, postfix=pid):
        if not preprocess_single_t1w(img_path, out_dir, ref_template, ref_crop):
            failed_id.append(img_path)
    print('Preprocessing failed in {} case(s)'.format(len(failed_id)))
    print(failed_id)




def get_opts():

    parser = argparse.ArgumentParser(description='t1w image preprocessing')

    ## Paths
    parser.add_argument('--img-list', type=str, default='../data',
                        help='csv file contains t1w image path')
    parser.add_argument('--ref-dir', type=str, default='../feat',
                        help='reference image for preprocessing')
    parser.add_argument('--out-dir', type=str, default='.',
                        help='output folder')
    parser.add_argument('--np', type=int, default=24,
                        help='number of processors')
    parser.add_argument('--part', type=int, default=0,
                        help='part')
    ##
    opts = parser.parse_args()
    return opts


def main():

    args = get_opts()

    ref_template = join(args.ref_dir, 'mni_icbm152_t1_tal_nlin_sym_09c.nii')
    ref_crop = join(args.ref_dir, 'ref_cropped_template.nii.gz')
    out_dir = args.out_dir
    makedirs(out_dir, exist_ok=True)

    print('Parent process %s.' % os.getpid())
    # img_list = list(pd.read_csv(args.img_list)['Id'])
    from glob import glob 
    img_list = sorted(glob(args.img_list+'ADNI/*/*/*/*/*'))
    L = len(img_list)//4
    if args.part == 0:
        img_list = img_list[:L]
    elif args.part == 1:
        img_list = img_list[L:2*L]
    elif args.part == 2:
        img_list = img_list[2*L:3*L]
    elif args.part == 3:
        img_list = img_list[3*L:]
    print('{} images'.format(len(img_list)))
    img_splits = np.array_split(img_list, args.np)
    assert sum([len(v) for v in img_splits]) == len(img_list)
    p = Pool(args.np)
    for i, split in enumerate(img_splits):
        result = p.apply_async(
            preprocess_t1w, args=(str(i), list(split), out_dir, ref_template, ref_crop)
        )
        result.get()
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

if __name__ == '__main__':
    main()


