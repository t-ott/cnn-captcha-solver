import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from .segmenter import Segmenter

class Evaluator:
    '''
    Class to evaluate the segmentation system and trained model on the test set
    of CAPTCHA images

    Attributes
    ----------
    Model : .model.Model
        PyTorch model, instance of the .model.Model class
    transform : PyTorch transforms
        PyTorch transforms chained together with torchvision.transforms.Compose()
    Segmenter : .segmenter.Segmenter
        Instance of the .segmenter.Segmenter class to segment characters from
        CAPTCHAs
    label_dict : dict
        Path to PyTorch state_dict file (typically .pt or .pth extension) to
        load the trained state of a model

    Methods
    -------
    predict():
        Use trained model to make a prediction on a CAPTCHA image
    evaluate_test_set():
        Evalute trained model on a test set of CAPTCHA images
    '''
    def __init__(self, Model, transform, label_key, model_state_dict=None):
        self.Model = Model
        if model_state_dict:
            self.Model.load_state_dict(torch.load(model_state_dict))
        self.Model.eval()

        self.transform = transform
        self.Segmenter = Segmenter()

        # Read in label_key CSV file that was made during data generation
        label_key_df = pd.read_csv(label_key)
        # Convert to dictionary with (int: char) items
        self.label_dict = pd.Series(
            label_key_df.char.values,
            index=label_key_df.int
        ).to_dict()


    def predict(self, img_path, print_results=False, plot_segmentation=False):
        '''Use trained model to make a prediction on a CAPTCHA image'''
        # Segment CAPTCHA image into characters
        segmented_chars = self.Segmenter.segment_chars(img_path, plot=plot_segmentation)

        # Predict on each character
        predictions = []
        for segmented_char in segmented_chars:
            x, y = segmented_char
            x = Image.fromarray(x) # to PIL image
            x = self.transform(x) # Apply Dataset transformations
            prediction = self.Model(x.unsqueeze(0))
            y_hat = torch.max(prediction, 1)[1].data.squeeze()
            # Convert integer prediction to character using label dict
            y_hat = self.label_dict[y_hat.item()]

            if print_results:
                print(f'Predicted: {y_hat} | Actual: {y}')

            predictions.append((y_hat, y))

        return predictions


    def evaluate_test_set(self, src_dir):
        '''Evalute trained model on a test set of CAPTCHA images'''
        test_img_paths = [os.path.join(src_dir, img_path)
                          for img_path in os.listdir(src_dir)
                          if img_path.lower().endswith(('.jpg', '.png'))]

        captcha_predictions = []
        char_predictions = []

        for test_img_path in test_img_paths:
            predictions = self.predict(test_img_path)
            char_predictions.extend(predictions)

            predicted_captcha = [''.join(chars[0] for chars in predictions)]
            actual_captcha = [''.join(chars[1] for chars in predictions)]
            captcha_predictions.append((predicted_captcha, actual_captcha))

        return EvaluatorResults(char_predictions, captcha_predictions, self.label_dict)


class EvaluatorResults:
    '''
    Results object returned by the Evaluator.evaluate_test_set() method

    Attributes
    ----------
    char_predictions : list
        List of tuples with (predicted character, actual character)
    captcha_predictions : list
        List of tuples with (predicted captcha text, actual captcha text)
    label_dict : dict
        Dictionary which contains integer character label keys, and alphanumeric
        characters (str) as values

    Methods
    -------
    plot_confusion_matrix():
        Plot a confusion matrix showing prediction results for every character
        in the test set
    '''
    def __init__(self, char_predictions, captcha_predictions, label_dict):
        self.char_predictions = char_predictions
        self.captcha_predictions = captcha_predictions
        self.label_dict = label_dict

        correct_chars = 0
        for chars in char_predictions:
            y_hat, y = chars
            if y_hat == y:
                correct_chars += 1
        self.char_accuracy = correct_chars / len(char_predictions)

        correct_captchas = 0
        for captchas in captcha_predictions:
            y_hat, y = captchas
            if y_hat == y:
                correct_captchas += 1
        self.captcha_accuracy = correct_captchas / len(captcha_predictions)


    def plot_confusion_matrix(self):
        '''
        Plot a confusion matrix showing prediction results for every character
        in the test set
        '''
        n_classes = len(self.label_dict)

        # pandas df with rows as predicted, cols as actual. initialize counts
        # with zeros
        conf_mat = pd.DataFrame(0, columns=self.label_dict.values(),
                                index=self.label_dict.values())

        for chars in self.char_predictions:
            y_hat, y = chars
            conf_mat.at[y_hat, y] += 1

        # create custom colormap for plot
        max_count = conf_mat.to_numpy().max()
        colors = cm.get_cmap('Wistia', max_count+1)
        new_colors = colors(np.arange(0, max_count+1))
        # change 0 counts to white to make matrix easier to interpret
        new_colors[0, :] = np.array([1, 1, 1, 1])
        new_cmap = ListedColormap(new_colors)

        fig, ax = plt.subplots(figsize=(12,12))
        conf_mat_plot = ax.matshow(conf_mat, cmap=new_cmap)

        # Place text with count in each cell
        for (i, j), count in np.ndenumerate(conf_mat):
            ax.text(j, i, count, ha='center', va='center')

        ax.set_xticks(np.arange(0, n_classes, 1))
        ax.set_xticklabels(list(self.label_dict.values()))
        ax.set_yticks(np.arange(0, n_classes, 1))
        ax.set_yticklabels(list(self.label_dict.values()))
        # Minor ticks, used to generate grid lines around each cell
        ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
        ax.grid(which='minor', linestyle='-')

        cbar = plt.colorbar(conf_mat_plot, fraction=0.04, pad=0.05)

        ax.set_xlabel('Actual')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Predicted')
        ax.set_title('Character Classification: Confusion Matrix', pad=15)

        plt.show()
