from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from model import create_model
from utils import Averager
from tqdm.auto import tqdm
from datasets import train_loader, valid_loader
import torch
import matplotlib.pyplot as plt
import time
import csv

plt.style.use('ggplot')

# Initialize the model and move to the computation device
model = create_model(num_classes=NUM_CLASSES)
model = model.to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad] # Get parameters of model that need to be updated
optimiser = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005) # Define the optimiser (used to update model parameters)

# Averager classes store the history of each epoch
train_loss_hist = Averager()
val_loss_hist = Averager()
train_itr = 1
val_itr = 1
# Train and validation loss lists to store loss values of all iterations
train_loss_list = []
val_loss_list = []

# function for running training iterations
def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for data in prog_bar:
        optimiser.zero_grad() # tell PyTorch to reset the gradients
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimiser.step()
        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list

# function for running validation iterations
def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list
    
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for data in prog_bar:
        images, targets = data
        images = [image.to(DEVICE) for image in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad(): # tell PyTorch to not calculate gradients as we are not training
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list

if __name__ == '__main__':
    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        # Reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()

        # Create two subplots for training and validation plots
        fig_1, train_ax = plt.subplots()
        fig_2, valid_ax = plt.subplots()

        # Start time and run the training and validation steps
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        
        # Every n epochs, save the loss plots and model
        if (epoch + 1) % SAVE_PLOTS_EPOCH == 0:
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            fig_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
            fig_2.savefig(f"{OUT_DIR}/valid_loss_{epoch+1}.png")
            print('SAVING PLOTS COMPLETE...')
            
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
            print('SAVING MODEL COMPLETE...\n')
        
    # Save loss plots and model once at the end
    train_ax.plot(train_loss, color='blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    valid_ax.plot(val_loss, color='red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')
    fig_1.savefig(f"{OUT_DIR}/results/train_loss_{epoch+1}.png")
    fig_2.savefig(f"{OUT_DIR}/results/valid_loss_{epoch+1}.png")
    torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
        
    plt.close('all')
    print('TRAINING COMPLETE')


# Save the training and validation loss values to a csv file
epochs = [i for i in range(1, NUM_EPOCHS+1)]
csv_header = ["Epoch", "Train loss", "Validation loss"]
csv_data = list(zip(epochs, train_loss_list, val_loss_list))
with open(f"{OUT_DIR}/results.csv", 'w') as f:
  writer = csv.writer(f)
  writer.writerow(csv_header)

  for row in csv_data:
    writer.writerow(row)