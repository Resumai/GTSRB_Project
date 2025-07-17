import torch.nn as nn
import torch.nn.functional as F
from configs import cnn_config as cfg


# SimpleCNN, is in fact, not that simple at all.
# I implemented this CNN with help, to better understand how layers interact. 
# I focused on grasping the data flow and transformation.
# While I certainly don’t yet fully understand every low-level mathematical detail, 
# I documented and annotated the structure as I learned.
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=cfg.NUM_CLASSES):  # GTSRB has 43 traffic sign classes
        super(SimpleCNN, self).__init__()

        # 3 input channels(rgb), 
        # 32 output filters/mini-detectors (feature maps),
        # size of detector window(3=3x3),
        # padding=1 helps keep image size same after this step
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # Input will become 32 channels(feature maps) from first layer(conv1)
        # and returns/outputs 64 new feature maps,
        # size of detector window(3=3x3),
        # padding=1 helps keep image size same after this step,
        # Now it looks for combinations of earlier features, like circle inside triangle or arrow shapes.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Downsample image by factor of 2 (height and width), keeps most important info.
        # Helps reduce computation and overfitting.
        self.pool = nn.MaxPool2d(2, 2) 


        # After 2 pooling layers, original 32x32 image becomes 8x8 with 64 channels.
        # So total features going into the linear layer = 64 * 8 * 8
        self.fc1 = nn.Linear(64 * 8 * 8, 128) # Fully connected layer
        self.fc2 = nn.Linear(128, num_classes) # Final output: Take 128 features and assign into one of 43 predictions. 

    def forward(self, x):
        # conv1(convulution) -> relu -> pool(shrink) = detect edges, shrink image, now we have 32 feature maps at 16x16
        x = self.pool(F.relu(self.conv1(x)))

        # conv2 -> relu -> pool = detect edges, shrink image again, now 64 feature maps at 8x8
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the 3D tensor to 1D per sample, so we can feed it into linear layer. Turn image into list of numbers.
        x = x.view(x.size(0), -1)

        # First dense layer: turn big image feature vector into smaller 128 features
        x = F.relu(self.fc1(x)) # Start reasoning

        # Final dense layer: produce raw class scores (logits), not softmax here
        x = self.fc2(x) # Final prediction (what sign it is?)

        # Output goes to loss function (e.g. CrossEntropyLoss)
        return x                              