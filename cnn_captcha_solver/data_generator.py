import os
import shutil
import random
import pandas as pd
import cv2

from .segmenter import Segmenter

class DataGenerator:
    '''
    Data generator that takes a directory of CAPTCHA images and spits them out
    into directories for training and testing sets

    Parameters
    ----------
    src_dir : str
        path to directory that contains CAPTCHA images, image files must be
        named with their CAPTCHA text
    random_seed : int, default=1
        integer seed to set random.seed()
    train_size : float, default=0.75
        size of training set to be extracted, range from 0.0 - 1.0

    Attributes
    ----------
    train_img_paths : list
        List of paths to character training images
    test_img_paths : list
        List of paths to CAPTCHA test images
    label_dict : dict
        Dictionary containing (alphanumeric character: integer label) items for
        each character in the training set
    Segmenter : .segmenter.Segmenter
        Instance of the .segmeneter.Segmenter class to segmenting test images

    Methods
    -------
    extract_train_set(target_dir, train_annotation_file='train_annotations.csv'):
        Uses Segmenter class to extract individual characters from directory of
        CAPTCHA images
    save_test_set(target_dir):
        Copy over test set images to target directory
    save_label_dict(target_path):
        Save label dictionary of integer labels of alphanumeric characters
        within CAPTCHA images to a CSV
    '''
    def __init__(self, src_dir, train_size=0.75, random_seed=None):
        if random_seed:
            random.seed(random_seed)

        img_paths = [
            os.path.join(src_dir, path) for path in os.listdir(src_dir)
            if path.lower().endswith(('.jpg', '.png'))
        ]

        random.shuffle(img_paths)
        split_index = int(len(img_paths) * train_size)
        self.train_img_paths = img_paths[:split_index]
        self.test_img_paths = img_paths[split_index:]

        # generate list of all characters in CAPTCHA images
        all_chars = []
        for img_path in img_paths:
            chars = os.path.basename(img_path).split('.')[0]
            for char in chars:
                all_chars.append(char)
        unique_chars = sorted(list(set(all_chars)))
        # assign integer label to each unique character
        self.label_dict = {char: i for i, char in enumerate(unique_chars)}

        self.Segmenter = Segmenter()


    def extract_train_set(self, target_dir, train_annotation_file='train_annotations.csv'):
        '''
        Uses Segmenter class to extract individual characters from directory of
        set CAPTCHA images, these char images serve as training data for the
        character classification model

        Parameters
        ----------
        target_dir : str
            Path for new directory, or empty directory, to contain character
            training images extracted from the trianing CAPTCHA images
        train_annotation_file : str, default=train_annotations.csv
            Path to annotations CSV file of training data, with each row
            containing (filename, label)
        '''

        if os.path.isdir(target_dir) == False:
            os.mkdir(target_dir)
        elif len([f for f in os.listdir(target_dir)
                  if f.lower().startswith(('.png', '.jpg'))]) > 0:
            raise ValueError(
                'target_dir needs to be a new directory or an existing directory'
                'free of image files'
            )

        # initialize list that will collect list of dictionaries to be record
        # annotations for each training example
        annotation_rows = []

        print(f'Segmenting chars from {len(self.train_img_paths)} images...')

        i = 0
        for img_path in self.train_img_paths:
            segmented_chars = self.Segmenter.segment_chars(img_path)

            for segmented_char in segmented_chars:
                if i % 2000 == 0:
                    print(f'Working on char {i}...')

                char_img, label = segmented_char
                # generate arbitrary numeric filename for each character
                target_img_fn = str(i).zfill(6)+'.png'
                try:
                    # write segmented character image to target directory
                    cv2.imwrite(os.path.join(target_dir, target_img_fn), char_img)
                    # use char label to lookup corresponding integer label
                    int_label = self.label_dict[label]
                    annotation_rows.append({'filename': target_img_fn, 'label': int_label})
                    i += 1
                except cv2.error as e:
                    print('\n*************************'
                          '\nFailed to write image with cv2.error:')
                    print(e)
                    continue

        annotation_df = pd.DataFrame(annotation_rows)
        annotation_df.to_csv('train_annotations.csv', index=False)
        print(f'Done! Saving filename/numeric class label key to file: {train_annotation_file}')


    def save_test_set(self, target_dir: str):
        '''
        Copy over test set images to target directory
        These will remain unsegmented/unaltered so the model can be evaluated on
        entirely unseen data
        '''
        if os.path.isdir(target_dir) == False:
            os.mkdir(target_dir)
        elif len([f for f in os.listdir(target_dir)
                  if f.lower().startswith(('.png', '.jpg'))]) > 0:
            raise ValueError(
                'target_dir needs to be a new directory or an existing directory'
                'free of image files'
            )

        print(f'Copying {len(self.test_img_paths)} images to {target_dir}...')

        for img_path in self.test_img_paths:
            target_path = os.path.join(target_dir, os.path.basename(img_path))
            shutil.copy(img_path, target_path)

        print(f'Copied {len(self.test_img_paths)} images to {target_dir}')


    def save_label_dict(self, target_path):
        '''
        Save label dictionary of integer labels of alphanumeric characters
        within CAPTCHA images to a CSV
        '''
        label_dict_items = self.label_dict.items()
        label_list = list(label_dict_items)
        label_df = pd.DataFrame(label_list, columns=['char', 'int'])
        label_df.to_csv(target_path, index=False)
