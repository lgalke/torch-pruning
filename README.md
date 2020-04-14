# torch-pruning
Pruning methods for pytorch with an optimizer-like interface

## usage

```python
import torch
import pruning

net = # an arbitrary pytorch nn.Module instance
dataloader = # some pytorch dataloader instance

optimizer = torch.optim.SGD(net.parameters(), 0.01, weight_decay=1e-5)
# Init pruning method in the same way as optimizer
pruning = pruning.MagnitudePruning(net.parameters(), 0.1, local=True,
                                   exclude_biases=True)

# Save initial parameters for later
w_0 = pruning.clone_params()

def train(net, dataloader, n_epochs=1):
    # Some standard training loop ...
    for epoch in range(n_epochs):
        for x, y in dataloader:
            # Do actually set *pruned* weights to zero
            pruning.zero_params()
            y_hat = net(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    

# Train epoch
train(net, dataloader, n_epochs=100)
# Do prune!
pruning.step()
# Rewind parameters to their values at init
pruning.rewind(w_0)
# Train the pruned model
train(net, dataloader, n_epochs=100)

# Do train and prune in a for loop, and then
# check if you have found a winning ticket
# ...

```
