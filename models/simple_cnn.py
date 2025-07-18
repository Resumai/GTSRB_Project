import torch.nn as nn
import torch.nn.functional as F
from configs import cnn_config as cfg


# SimpleCNN, is in fact, not that simple at all.
# I implemented this CNN with help, to better understand how layers interact. 
# I focused on grasping the data flow and transformation.
# While I certainly don’t yet fully understand every low-level mathematical detail, 
# I documented and annotated the structure as I learned.
class SimpleCNN(nn.Module):
    def __init__(self, 
                num_classes=cfg.NUM_CLASSES, # GTSRB has 43 traffic sign classes
                use_dropout=cfg.USE_DROPOUT, 
                dropout_rate=cfg.DROPOUT_RATE,
                cnn_depth=cfg.CNN_DEPTH):

        super(SimpleCNN, self).__init__()
        self.use_dropout = use_dropout

        if cnn_depth is None:
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
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(128, num_classes) # Final output: Take 128 features and assign into one of 43 predictions. 

            self.forward_mode
        else: # dynamic layer formation for experimentation, my brain is not braining this part properly.
            filters = cfg.CNN_FILTERS
            in_channels = 3
            layers = []

            for i in range(cnn_depth):
                out_channels = filters[i % len(filters)]
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                in_channels = out_channels

            self.feature_extractor = nn.Sequential(*layers)

            final_size = 32 // (2 ** cnn_depth)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels * final_size * final_size, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, num_classes)
            )

            self.forward_mode = "dynamic"

    def forward(self, x):
        if self.forward_mode == "fixed":
            # conv1(convulution) -> relu -> pool(shrink) = detect edges, shrink image, now we have 32 feature maps at 16x16
            x = self.pool(F.relu(self.conv1(x)))

            # conv2 -> relu -> pool = detect edges, shrink image again, now 64 feature maps at 8x8
            x = self.pool(F.relu(self.conv2(x)))

            # Flatten the 3D tensor to 1D per sample, so we can feed it into linear layer. Turn image into list of numbers.
            x = x.view(x.size(0), -1)

            # First dense layer: turn big image feature vector into smaller 128 features
            x = F.relu(self.fc1(x)) # Start reasoning
            

            if self.use_dropout:
                x = self.dropout(x)

            # Final dense layer: produce raw class scores (logits), not softmax here
            x = self.fc2(x) # Final prediction (what sign it is?)
        else: # Dynamic depth
            x = self.feature_extractor(x)
            x = self.classifier(x)
        # Output goes to loss function (e.g. CrossEntropyLoss)
        return x                              