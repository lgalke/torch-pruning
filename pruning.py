import torch
import torch.nn as nn
import torch.nn.functional as F


def cut(pruning_rate, flat_params):
    """ Compute cutoff value within `flat_params` at percentage `pruning_rate`."""
    assert flat_params.dim() == 1
    # Compute cutoff value
    with torch.no_grad():
        cutoff_index = round(pruning_rate * flat_params.size()[0])
        values, __indices = torch.sort(torch.abs(flat_params))
        cutoff = values[cutoff_index]
    return cutoff


class MagnitudePruning():
    """ Magnitude pruning with an optimizer-like interface  """

    def __init__(self, params, pruning_rate=0.25, local=True,
                 exclude_biases=False):
        """ Init pruning method """
        self.local = bool(local)
        self.pruning_rate = float(pruning_rate)

        if exclude_biases:
            # Discover all non-bias parameters
            self.params = [p for p in params if p.dim() > 1]
        else:
            self.params = [p for p in params]

        # init masks to all ones
        masks = []
        for p in self.params:
            masks.append(torch.ones_like(p))

        self.masks = masks

    ################################################
    # Reporting nonzero entries and number of params
    def count_nonzero(self):
        """ Count nonzero elements """
        return int(sum(mask.sum() for mask in self.masks).item())

    def numel(self):
        """ Number of elements """
        return int(sum(mask.view(-1).size(0) for mask in self.masks))
    ################################################

    ############################################
    # Methods for resetting or rewinding params
    def clone_params(self):
        """ Copy all tracked params, such that they we can rewind to them later """
        return [p.clone() for p in self.params]

    def rewind(self, cloned_params):
        """ Rewind to previously stored params """
        for p_old, p_new in zip(self.params, cloned_params):
            p_old.data = p_new.data
    ############################################

    ##############
    # Core methods
    def step(self):
        """ Update the pruning masks """
        if self.local:  # Local (layer-wise) pruning #
            for i, (m, p) in enumerate(zip(self.masks, self.params)):
                # Compute cutoff
                flat_params = p[m == 1].view(-1)
                cutoff = cut(self.pruning_rate, flat_params)
                # Update mask
                new_mask = torch.where(torch.abs(p) < cutoff,
                                       torch.zeros_like(p), m)
                self.masks[i] = new_mask
        else:  # Global pruning #

            # Gather all masked parameters
            flat_params = torch.cat([p[m == 1].view(-1)
                                     for m, p in zip(self.masks, self.params)])
            # Compute cutoff value
            cutoff = cut(self.pruning_rate, flat_params)

            # Calculate updated masks
            for i, (m, p) in enumerate(zip(self.masks, self.params)):
                new_mask = torch.where(torch.abs(p) < cutoff,
                                       torch.zeros_like(p), m)
                self.masks[i] = new_mask

    def zero_params(self, masks=None):
        """ Apply the masks, ie, zero out the params """
        masks = masks if masks is not None else self.masks
        for m, p in zip(masks, self.params):
            p.data = m * p.data
    ##############
