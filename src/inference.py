import numpy as np
import cv2
import os
import torch
import glob as glob
from model import create_model
import albumentations as A
# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes=4).to(device)
model.load_state_dict(torch.load(
    '../outputs/model4.pth', map_location=device #TODO: change the path to the latest model file
))
model.eval()

# directory where all the images are present
SEPARATOR = os.path.sep
TEST_DATA_DIR = '../test_data'
TEST_PREDICTIONS_DIR = '../test_predictions'
test_images = glob.glob(f"{TEST_DATA_DIR}/*")
print(f"Test instances: {len(test_images)}")
# classes: 0 index is reserved for background
CLASSES = [
    'background', 'apple', 'banana', 'orange'
]
# Create output folders if they don't exist
if not os.path.exists(TEST_PREDICTIONS_DIR):
    os.makedirs(TEST_PREDICTIONS_DIR)

# Threshold for bounding boxes
detection_threshold = 0.8 # TODO: increase threshold to 0.8 when model gets better

for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split(SEPARATOR)[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    if torch.cuda.is_available():
        image = torch.tensor(image, dtype=torch.float).cuda()
    else:
        image = torch.tensor(image, dtype=torch.float)
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)
    
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        print("Number of predictions:", len(boxes))
        print(f"Scores max: {scores.max()}, min: {scores.min()}, mean: {scores.mean()}")
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        print("Number of predictions after thresholding:", len(boxes))
        draw_boxes = boxes.copy()
        # get all the predicted class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 0, 255), 2)
            cv2.putText(orig_image, pred_classes[j], 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                        2, lineType=cv2.LINE_AA)
        # cv2.imshow('Prediction', orig_image)
        # cv2.waitKey(1)
        filename = TEST_PREDICTIONS_DIR + SEPARATOR + image_name + "_prediction.jpg"
        cv2.imwrite(filename, orig_image)
    print(f"Image {i+1} done...")
    print('-'*50)
print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()