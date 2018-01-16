"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
import cv2
from torch.nn import ReLU

from misc_functions import (get_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            show_gradient_image,
                            get_positive_negative_saliency)

# TODO:
# - add notes
# - add support for backprop from intermediate layers
# - refactor the interface to increase flexibility
# - add activation visualization
# - add filter visualization for first several layers
class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the
       target_layer to the given image.
    """
    def __init__(self, model, processed_im, target_class, target_layer=19):
        self.model = model
        self.input_image = processed_im
        self.target_class = target_class
        self.target_layer = target_layer
        self.gradients = None
        self._target_layer = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers(target_layer)

    def hook_layers(self, target_layer):
        def hook_function(module, grad_in, grad_out):
            """save input space gradient"""
            self.gradients = grad_in[0]

        def forward_hook(module, input, output):
            """save output of a certain layer"""
            self._target_layer = output
        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

        # test register forward hook
        tgt = list(self.model.features._modules.items())[36][1]
        tgt.register_foward_hook(forward_hook())


    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self):
        # Forward pass
        model_output = self.model(self.input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][self.target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


    def generate_gradients(self, layer=19):
        """
        Visualize which part of the image activates the neurons in layer the most.
        For those layers that have enormous filters, a random filter is selected
        for observation.

        This module uses VGG19 (config 'E') as the model.
        :param layer: the number of layer on which the guided back propagation starts.
        :return: the gradients in Input domain, as in numpy array format.
        """
        # Forward pass
        self.model(self.input_image)
        # Zero gradients
        self.model.zero_grad()
        # test layer 16 relu (1, 512, 14, 14)
        one_hot_output = torch.FloatTensor(self._target_layer.size()).zero_()
        one_hot_output[0, 0, :, :] = 1
        # Backward pass
        self._target_layer.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

if __name__ == '__main__':
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example)

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model, prep_img, target_class)
    # Get gradients
    guided_grads = GBP.generate_gradients()
    show_gradient_image(guided_grads, "Original")
    # Save colored gradients
    #save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    show_gradient_image(grayscale_guided_grads, "Grayscale")
    # Save grayscale gradients
    #save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    show_gradient_image(pos_sal, "Positive saliency")
    show_gradient_image(neg_sal, "Negative saliency")
    #save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    #save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    cv2.waitkey(0)
    print('Guided backprop completed')
