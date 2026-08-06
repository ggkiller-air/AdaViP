# AdaViP package

The package currently contains the data-format-independent AdaViP perception
and policy core.

## Model boundary

`AdaViPPerception` receives modality-specific feature sequences, a task
embedding, and an optional progress context. It performs:

1. HyperNet-generated low-rank residual transforms for each modality;
2. shared task-invariant decoders;
3. cross-modal attention at each timestep;
4. a HyperNet-generated fusion transform.

`AdaViPPolicy` then forwards the fused representation to an injected action
backbone. Frozen base encoders can be injected without coupling the policy to
HDF5, LeRobot, Zarr, or any other collection format.

Data readers, synchronization, conversion, and temporal sampling belong in a
future `adavip/processing/` package. Raw datasets remain outside this
repository.
