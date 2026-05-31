# Remote Sensing Landcover Classification

This repository is my solution for the first MLESS homework assignment. The task is to classify small SAT-6 satellite image patches into six landcover classes with two approaches:

- Random Forest classifier from scikit-learn
- Convolutional Neural Network in PyTorch

## Repository Contents

| File | Purpose |
| --- | --- |
| `Random_forest_classifier_on_remote_sensing_image.ipynb` | Random Forest baseline, channel inspection, channel experiments, per-class metrics, examples, and hyperparameter test |
| `CNN_classifier_on_remote_sensing_image.ipynb` | CNN baseline, RGB-only experiment, per-class metrics, examples, and learning-rate test |
| `README.md` | Documentation of the assignment, data, methods, results, and answers |
| `requirements.txt` | Python packages needed to run the notebooks |
| `.gitignore` | Keeps the large local `data/` folder out of git |

## Assignment Coverage

The model and documentation tasks from sections 2 and 3 are covered as follows:

| Task | Status | Where it is addressed |
| --- | --- | --- |
| 2.1 Run the Random Forest notebook and answer its questions | Done | Random Forest notebook plus "Random Forest Notebook Questions" below |
| 2.2 Add per-class accuracy and correct/wrong examples for Random Forest | Done | Random Forest notebook, confusion matrix, per-class table, example plots |
| 2.3 Inspect R, G, B, NIR and rerun Random Forest with RGB only | Done | "Channel Inspection" and "Random Forest Results" |
| 2.4 Rerun Random Forest with R, G, NIR | Done | "Random Forest Results" |
| 2.5 Change one Random Forest hyperparameter and justify it | Done | `max_depth=20` experiment |
| 3.1 Run and understand the CNN notebook | Done | CNN notebook executed from start to finish |
| 3.2 Add per-class accuracy and correct/wrong examples for CNN; document changes from 2.2 | Done | "CNN Method Notes" and "CNN Results" |
| 3.3 Rerun CNN with RGB only and compare degradation | Done | "CNN Results" and comparison text |
| 3.4 Change one CNN hyperparameter and justify it | Done | `learning_rate=0.0005` experiment |

The GitHub submission tasks are process steps outside the model notebooks. This README focuses on the code, experiments, and documented answers.

## Data

The data is a subset of the SAT-6 dataset. Each row in `X_test_sat6.csv` is one flattened `28 x 28` image patch with four channels:

- red
- green
- blue
- near infrared (NIR)

So each sample has `28 x 28 x 4 = 3136` feature values. The labels in `y_test_sat6.csv` are one-hot encoded and correspond to the class names in `sat6annotations.csv`.

Download the data files into a local `data` folder:

```bash
mkdir -p data
wget -P data https://b2share.eudat.eu/api/files/a697daf7-7570-44ff-854c-0fab43f2b52c/X_test_sat6.csv
wget -P data https://b2share.eudat.eu/api/files/a697daf7-7570-44ff-854c-0fab43f2b52c/y_test_sat6.csv
wget -P data https://b2share.eudat.eu/api/files/a697daf7-7570-44ff-854c-0fab43f2b52c/sat6annotations.csv
```

The available data is imbalanced:

| Class | Samples |
| --- | ---: |
| building | 3,714 |
| barren_land | 18,367 |
| trees | 14,185 |
| grassland | 12,596 |
| road | 2,070 |
| water | 30,068 |

For all experiments I used a balanced subset with 1,000 training samples and 100 test samples per class. The split uses `RANDOM_STATE = 42`, and both notebooks verify that train/test overlap is `0`.

## Setup

Install the Python dependencies:

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
python -m ipykernel install --user --name mless-homework --display-name "MLESS Homework"
jupyter notebook
```

Then run the notebooks in this order:

1. `Random_forest_classifier_on_remote_sensing_image.ipynb`
2. `CNN_classifier_on_remote_sensing_image.ipynb`

## Experimental Setup

The same balanced train/test split is used for both model families. The main evaluation metric is overall accuracy, but I also report per-class accuracy because the classes behave differently.

For each model I saved:

- overall accuracy
- per-class accuracy
- confusion matrix
- examples of correct classifications
- examples of misclassifications

The example plots show the original sample index, the true class, and the predicted class.

## Channel Inspection

Before removing channels, I inspected the four input channels in the Random Forest notebook. The notebook shows one sample as an RGB image and then as four separate grayscale channel images.

I also calculated the mean channel intensity per class:

| Class | R | G | B | NIR |
| --- | ---: | ---: | ---: | ---: |
| building | 188.3 | 187.4 | 195.8 | 122.6 |
| barren_land | 173.5 | 158.5 | 136.7 | 180.5 |
| trees | 91.0 | 92.8 | 84.5 | 145.1 |
| grassland | 118.5 | 117.3 | 101.6 | 171.1 |
| road | 152.5 | 153.0 | 151.8 | 87.2 |
| water | 70.6 | 85.8 | 107.3 | 17.0 |

The NIR channel is especially different for vegetation and water. Trees and grassland have high NIR means, while water has a very low NIR mean. This is why I expected NIR to be useful for classification, even though the final accuracy differences are small on this particular split.

## Random Forest Notebook Questions

| Question | Answer |
| --- | --- |
| How can I extract only green and infrared? | Reshape the flat data to `(-1, 28, 28, 4)`, select channels `[1, 3]`, and flatten again if the classifier expects a 2D feature matrix. |
| Why is one-hot encoding useful? | It makes the class order explicit and represents each class in a separate column. For scikit-learn metrics I convert it back to one class id with `argmax`. |
| Why use `extend` here and `append` above? | `append` adds a whole list as one nested element. `extend` adds the selected indices one by one into the existing list. |
| What was wrong with the original train/test sampling? | Train and test indices were sampled independently, so the same sample could appear in both sets. I changed the split to shuffle once per class and then take separate slices. |
| Why shuffle train and test samples? | Shuffling avoids class-ordered data and makes later training or batching less dependent on the order in which classes were selected. |

## Random Forest Results

The Random Forest uses flattened pixel values as features. For the channel experiments, I first reshape the data back to image form, select the required channels, and flatten it again.

| Experiment | Channels | Overall accuracy |
| --- | --- | ---: |
| Baseline | R, G, B, NIR | 0.942 |
| RGB only | R, G, B | 0.940 |
| R, G, NIR | R, G, NIR | 0.940 |
| Hyperparameter test | R, G, B, NIR, `max_depth=20` | 0.948 |

Per-class accuracy:

| Experiment | building | barren_land | trees | grassland | road | water |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.910 | 0.970 | 0.960 | 0.870 | 0.940 | 1.000 |
| RGB only | 0.910 | 0.950 | 0.990 | 0.910 | 0.910 | 0.970 |
| R, G, NIR | 0.900 | 0.960 | 0.960 | 0.900 | 0.920 | 1.000 |
| `max_depth=20` | 0.940 | 0.960 | 0.950 | 0.900 | 0.940 | 1.000 |

### Random Forest Interpretation

Removing NIR did not strongly reduce the overall Random Forest accuracy in this run. The full four-channel model scored `0.942`, while RGB only and R, G, NIR both scored `0.940`.

The class-level changes are more informative than the overall score. RGB only improved trees and grassland in this split, but water dropped from `1.000` to `0.970`. R, G, NIR kept water at `1.000`, but building and road were slightly worse than the full model.

For the hyperparameter task I changed `max_depth` to `20`. My expectation was that limiting tree depth could reduce overfitting and make the forest a little more robust, but it could also reduce accuracy if the trees became too simple. In this run, `max_depth=20` improved accuracy from `0.942` to `0.948`, so the depth limit did not hurt the model on this test split.

## CNN Method Notes

The CNN uses image tensors instead of flattened vectors. For task 3.2, the evaluation code is similar to the Random Forest evaluation, but the prediction step is different:

- the data must be read through a PyTorch `Dataset` and `DataLoader`
- model outputs are logits, so predictions are selected with `argmax`
- labels are class ids, not one-hot rows, because `CrossEntropyLoss` expects class ids

For the RGB-only CNN experiment, more code changes were needed than for Random Forest:

- the dataset returns only the selected channels
- normalization mean and standard deviation are recomputed for those channels
- the first convolution layer is created with the matching input channel count

## CNN Results

| Experiment | Channels | Overall accuracy |
| --- | --- | ---: |
| Baseline | R, G, B, NIR | 0.962 |
| RGB only | R, G, B | 0.952 |
| Hyperparameter test | R, G, B, NIR, `learning_rate=0.0005` | 0.958 |

Per-class accuracy:

| Experiment | building | barren_land | trees | grassland | road | water |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.960 | 0.970 | 0.960 | 0.920 | 0.960 | 1.000 |
| RGB only | 0.880 | 0.940 | 0.970 | 0.930 | 0.990 | 1.000 |
| `learning_rate=0.0005` | 0.960 | 0.950 | 0.980 | 0.890 | 0.970 | 1.000 |

### CNN Interpretation

The CNN degraded more clearly than the Random Forest when NIR was removed. The CNN changed from `0.962` with all four channels to `0.952` with RGB only. The largest drop was for building, from `0.960` to `0.880`.

This RGB-only effect is larger than in the Random Forest experiment, where the overall accuracy changed only from `0.942` to `0.940`. On this split, the CNN used the extra NIR information more effectively.

For the CNN hyperparameter task I changed the Adam learning rate from `0.001` to `0.0005`. I expected smaller updates to be more stable but slower within only 10 epochs. The result was slightly lower than the baseline, `0.958` instead of `0.962`, which agrees with the expectation.

## Correct And Wrong Classifications

Both notebooks plot correct and incorrect predictions after the metrics. The goal is not only to report a number, but also to see what kind of samples are difficult.

In the saved examples, mistakes mostly happen between visually similar classes, especially vegetation-related classes or built surfaces. Water is the easiest class in these runs and reaches `1.000` per-class accuracy in most experiments.

## Main Takeaways

- The CNN baseline performed best overall with accuracy `0.962`.
- Removing NIR had a small effect on Random Forest in this split, but a clearer effect on CNN.
- NIR is still meaningful in the data: the channel inspection shows strong class-level differences, especially for vegetation and water.
- The `max_depth=20` Random Forest experiment slightly improved the score.
- The smaller CNN learning rate was a little worse after 10 epochs, probably because training progressed more slowly.

## Limitations

These results are from one balanced split with 6,000 training samples and 600 test samples. The exact numbers can change with a different random split. A stronger evaluation would repeat the experiments over several seeds and report mean and standard deviation.

The CNN was trained for only 10 epochs on CPU. Longer training, data augmentation, or a separate validation set could change the final comparison.

## Author

Valeriia Iordanova

Master’s Program in Computer Science
University of Cologne
