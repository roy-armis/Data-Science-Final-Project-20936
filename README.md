The notebook is a bit cluttered as the course requirement was to have all stages and experiments in a single notebook.
Might refactor later and extract only the good stuff :)

It is also very large (contains many images), so Github won't render it. You can enjoy it in this [Google Colab notebook](https://colab.research.google.com/drive/1MTfxjAlY-_lecIdz27QEQ1i3feMRGqrn?usp=sharing)

<br>

# Humpback Whale Fluke Identification

This notebook was created by Roy Gonen and Amit Aharoni as part of the Data Science workshop at the Open University of Israel, Spring of 2022.

## Purpose and goal definition

whales play an important role in the marine ecosystem and their populations are under threat due to human activities such as pollution, overfishing, and habitat destruction. The hunting and killing of these animals can further harm their populations and disrupt the balance of the ecosystem. In addition, many of the methods used to hunt whales, such as harpoons and explosives, can cause unnecessary suffering and death for the animals, and harm the marine ecosystem as well.

Scientists and researches use photos of whale flukes (basically their tails) to mointor them, and help the populations recover. Whale flukes are unique and contain different shapes, colors and markings. For decades this work has been done manually by individuals, causing many data discrepancies, identification mistakes and mostly - very slow and inefficient work.

Happywhale is an organization that helps monitoring whales, and it gathered over 25,000 photos of whale flukes.
They published a Kaggle competition, to help building an algorithm to identify individual whales from photos.

## Challenges dealing with whale photos

The Happywhale dataset contains over 25,000 photos of whale flukes, taken by individual scientists/researchers, proffesional and amateur photographers over the globe.

The photos are not stardardized - they vary in equipment, photoshoot skills, weather, location (shore/boats) and timing, which makes it more difficult to work with.

In addition, most whales have one or just few photos labeled to them, making it hard to build a learning system that understands unique differences between flukes. The dataset is not balaced either, with over 38% of photos not being labeled.

## Solution evaluation

As defined by the Kaggle competition, out evaluation metric will be top 5 accuracy - meaning amount of photos that were labeled correctly from all the test photos, while allowing 5 different label guesses for each photos.

# Table of Contents

## 1. EDA

We started with some exploratory data analysis (EDA) to familiarize ourselves with the raw data, it’s characteristics and difficulties.
We soon realized that the images contain many types of noise such as bad angles, sketches, text over the photo and more.
In addition, It was clear that the dataset isn’t balanced, and the variance is too large to try and balance it.
Two more challenges that we have faced due to the data structure are:

- There are 5005 classes while one of them is new_whale, which can mean either no one ever saw this whale, or that it is not classified to its whale_id.
- Most of the whales had very low number of instances in the data, with over 2000 whales having only one image, making it difficult for a neural network to learn (ie separating to train/validation)
  We also observed that images can be either grayscale or RGB, and in various shapes.
  Applying some traditional Computer Vision algorithms verified something we had felt all along the EDA - we need to find a way to crop only the whale fluke from the image to remove noisy background.
![image](https://github.com/roy-armis/Data-Science-Final-Project-20936/blob/main/assets/bfmatcher-accurate-matches%20example.png)


## 2. Fluke Bounding Box Detection

We explored some Object Detection models, and decided to go with YOLOv7, which we trained on a set of bounding box annotated images, and then predicted our images’ bounding boxes.
We compared tout model results to a pertained zero-shot open-vocabulary model by Google, which had surprisingly good results, but not good enough for us to be confidence with.
![image](https://github.com/roy-armis/Data-Science-Final-Project-20936/blob/main/assets/whale-fluke-detection-example.png)

## 3. Model

Equipped with both good understanding of our data and its challenges, and with the ability to crop images correctly, we approached our main task - finding and training a model to classify whale fluke images to their individual whale ids.
Initially we tried a naive traditional approach using HOG feature descriptor to extract features from the image and then pass those features to an SVM classifier, which didn’t return satisfying results.

We soon realized that Deep Learning approaches would fit best to our problem, and specifically CNNs.
We explored and experimented many models and loss functions to find an approach that is similar to face recognition tasks - Arcface. 

After some backbone selection and hyperparameter optimization, we finally had our complete architecture - ArcFace with EfficientNetV2 as the backbone, and for testing using the trained embedding features for each class, then performing cosine similarity to predict the most suitable class.
During prediction we also had to address our new_whale challenge and we solved it by finding a threshold that marks when we can predict the class new_whale.
![image](https://github.com/roy-armis/Data-Science-Final-Project-20936/blob/main/assets/final-kaggle-score.png)

## 4. Performance

Lastly, we were curious to understand what our network has learnt and after sadly failing to use SHAP’s DeepExplainer, we decided to implement Activation Maps which helped us visualize the locations where the network focused and used to make it class prediction, such as special marks and fluke shapes.
![image](https://github.com/roy-armis/Data-Science-Final-Project-20936/blob/main/assets/activation-map-example.png)
