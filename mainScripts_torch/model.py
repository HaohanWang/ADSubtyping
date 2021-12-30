import torch.nn as nn
from dropblock import DropBlock3D, LinearScheduler

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class PadMaxPool3d(nn.Module):
    def __init__(self, kernel_size, stride, return_indices=False, return_pad=False):
        super(PadMaxPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool3d(kernel_size, stride, return_indices=return_indices)
        self.pad = nn.ConstantPad3d(padding=0, value=0)
        self.return_indices = return_indices
        self.return_pad = return_pad

    def set_new_return(self, return_indices=True, return_pad=True):
        self.return_indices = return_indices
        self.return_pad = return_pad
        self.pool.return_indices = return_indices

    def forward(self, f_maps):
        coords = [self.stride - f_maps.size(i + 2) % self.stride for i in range(3)]
        for i, coord in enumerate(coords):
            if coord == self.stride:
                coords[i] = 0

        self.pad.padding = (coords[2], 0, coords[1], 0, coords[0], 0)

        if self.return_indices:
            output, indices = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, indices, (coords[2], 0, coords[1], 0, coords[0], 0)
            else:
                return output, indices

        else:
            output = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, (coords[2], 0, coords[1], 0, coords[0], 0)
            else:
                return output


class Conv5_FC3(nn.Module):
    """
    Classifier for a binary classification task

    Subject level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5, dropblock=False, blocksize=10, 
        convKernel1=3, poolKernel1=3, convKernel2=3, poolKernel2=3, poolKernel3=3, 
        fcSize1=32256, fcSize2=1300):
        super(Conv5_FC3, self).__init__()

        if dropblock:    # is this correct? 
            print("training model with dropblock")
        self.dropblock = dropblock

        self.drop_block = LinearScheduler(  # what does this mean?
            DropBlock3D(drop_prob=dropout, block_size=blocksize),
            start_value=0.,
            stop_value=dropout,
            nr_steps=10
        )

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, convKernel1, padding=1), 
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(poolKernel1, poolKernel1), 

            nn.Conv3d(8, 16, convKernel2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(poolKernel2, poolKernel2),

            nn.Conv3d(16, 32, convKernel2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(poolKernel2, poolKernel2),

            nn.Conv3d(32, 64, convKernel2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(poolKernel3, poolKernel3),

            nn.Conv3d(64, 128, convKernel2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(poolKernel3, poolKernel3),

        )

        self.classifier_ori = nn.Sequential( 

            Flatten(),
            nn.Dropout(p=dropout),
            
            nn.Linear(fcSize1, fcSize2),
            nn.ReLU(),

            nn.Linear(fcSize2, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.classifier_dropblock = nn.Sequential(

            self.drop_block,
            Flatten(),
            
            nn.Linear(fcSize1, fcSize2),
            nn.ReLU(),

            nn.Linear(fcSize2, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward_ori(self, x):
        
        x = self.features(x)
        x = self.classifier_ori(x)

        return x
    
    def forward_dropblock(self, x):

        self.drop_block.step()
        x = self.features(x)
        x = self.classifier_dropblock(x)

        return x

    def forward(self, x):
        return self.forward_dropblock(x)
        # if self.dropblock:
        #
        # else:
        #     return self.forward_ori(x)

def get_model(config):
    print('model name:', config.model.name)
    f = globals().get(config.model.name)
    if config.model.params is None:
        return f()
    else:
        return f(**config.model.params)

