import torch
import torch.nn as nn
import numpy as np

TESTING=True
PRINT_SIZES = True

class TemplateMotionGeneration(nn.Module):
    def __init__(self, input_dim, passthrough_dim, layer_widths, dtype, use_softplus=False, use_elu=False, use_tanh = False,
                 use_custom_initialization = False,
                 # grad_t is the constant, relatively large, abs value of default slope of first 
                 # linear layer in the $t$ direction
                 # t_intercept_padding is padding such that the first linear layer
                 # has t-intercept uniformly distributed in the interval [-t_intercept_padding, 1+t_intercept_padding]
                 custom_initialization_grad_t = None,
                 custom_initialization_t_intercept_padding = None):
        super(TemplateMotionGeneration, self).__init__()
        self.input_dim = input_dim
        self.passthrough_dim = passthrough_dim
        self.layer_widths = layer_widths
        self.dtype = dtype

        self.nonlinearity = torch.nn.Softplus() if use_softplus else torch.nn.ELU() if use_elu else torch.nn.Tanh() if use_tanh else torch.nn.ReLU()

        # initial input has dimension input_dim + passthrough_dim 
        previous_layer_width = self.input_dim + self.passthrough_dim
        self.all_layers = []
        for i, w in enumerate(layer_widths):
            self.all_layers.append(nn.Linear(previous_layer_width, w))
            previous_layer_width = w

        # set to double as needed
        # maybe equivalent to self=self.double()?
        if dtype == torch.double:
            print("Setting to double")
            for i in range(len(self.all_layers)):
                self.all_layers[i] = self.all_layers[i].double()
                
        self.all_layers = nn.ModuleList(self.all_layers)

        # custom initialization
        self.custom_initialization_grad_t = custom_initialization_grad_t
        self.custom_initialization_t_intercept_padding = custom_initialization_t_intercept_padding
        if use_custom_initialization:
             self.initialize()

    def initialize(self):
      # assert that the caller set the needed parameters 
      assert self.custom_initialization_grad_t
      assert self.custom_initialization_t_intercept_padding 
      fan_out = self.all_layers[0].weight.shape[0] # type: int
      with torch.no_grad():
          sign = torch.randint(0,2,size=(fan_out,)) * 2 - 1
          # overwrite the initialization of the last column (the time weight)
          # so that it is +/- grad_t
          grad_t = sign * self.custom_initialization_grad_t 
          intercept = torch.rand(fan_out) * (1 + 2 * self.custom_initialization_t_intercept_padding) - self.custom_initialization_t_intercept_padding
          bias = - grad_t * intercept
          # maybe there's a better way to copy data into tensors,
          # but we note that when we try to do the below we do need to
          # have "bias[:] =..."
          # If you try just "bias = ..."
          # You get the error
          # TypeError: cannot assign 'torch.FloatTensor' as parameter 'bias' (torch.nn.Parameter or None expected)
          self.all_layers[0].weight[:,-1] = grad_t
          self.all_layers[0].bias[:] = bias
        
    def mean_square_layer_weights(self):
        sum_weights = 0
        count_weights = 0
        for layer in self.all_layers:
            sum_weights += torch.sum(torch.square(layer.weight))
            count_weights += np.product(np.array(layer.weight.shape)).item()
        return sum_weights/count_weights

    def forward(self, x, t):
        if TESTING:
            assert len(x.shape) == 2, "batch of x coords should be two dimensional"
            assert len(t.shape) == 2, "batch of t vals should be two dimensional"
            assert x.shape[0] == t.shape[0], "inputs should have same batchsize"
        layer = torch.hstack((x,t))
        if PRINT_SIZES:
          print("motion_model_input", layer.shape)
        for i, fc in enumerate(self.all_layers):
            if i != len(self.all_layers) - 1:
                layer = self.nonlinearity(fc(layer))
            else: # don't nonlinear the last layer
                layer = fc(layer)
            if PRINT_SIZES:
              print("after fc: ", layer.shape)
        return layer
