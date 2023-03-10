# cnn-fruit-detector
A convolutional neural network to detect apples, oranges and bananas

# Setup
Make sure to extract the data from the zip file and ensure that the folder structure is as follows:
```bash
├── data
│   ├── test
│   └── train
```

# Training
Currently only tested by running `fruit_detector.ipynb`. `src/engine.py` and `main.py` are not up-to-date

Reference: https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/
Dataset from: https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection

# Testing
Run `src/inference.py` and change the path of the model to the model you want to use.

You can also extract from [this](https://drive.google.com/file/d/1CaNJ-HbwGiXbuWLGAO7wY7zpb4vPQf6O/view?usp=share_link) zip file to use a pretrained model