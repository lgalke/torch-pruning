# torch-pruning
Pruning methods for pytorch with an optimizer-like interface

```python
import pruning

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), 0.01, weight_decay=1e-5)
# Init pruning method with model parameters
pruning = MagnitudePruning(net.parameters(), 0.1, local=True)


# Save initial parameters
w_0 = pruning.clone_params()

# Train one epoch
for x, y in train_loader:
    # Zero out parameters that are already pruned
    pruning.zero_params(masks)
    y_hat = net(x)
    loss = criterion(y_hat, y.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()

# Do prune!
pruning.step()
# Rewind the parameters to initalization values (as in the LTH)
pruning.rewind(w_0)

# Train the pruned model
# ...

```
