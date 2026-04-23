# Recipe: Custom Model Wrapper

When your model does not fit the standard `PyTorchModel` or
`TensorFlowModel` assumptions, write a custom `ModelWrapper` subclass.

## Minimal Subclass

```python
from quinkgl.models.base import ModelWrapper, TrainingConfig, TrainingResult

class MyCustomWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)

    def get_weights(self):
        # Return any serialisable dict / list.
        return {"params": self.model.params.copy()}

    def set_weights(self, weights):
        self.model.params = weights["params"].copy()

    async def train(self, data, config=None):
        # Train for one local epoch.
        self.model.fit(data)
        return TrainingResult(
            epochs_completed=1,
            final_loss=self.model.last_loss,
            samples_trained=len(data),
        )

    def evaluate(self, data, loss_fn=None):
        loss, acc = self.model.score(data)
        return {"loss": loss, "accuracy": acc}
```

## Hooking into `build_model`

Return your wrapper directly from the peer script:

```python
def build_model(manifest, **kwargs):
    internal = MyInternalModel()
    return MyCustomWrapper(internal)
```

The CLI detects that the returned object already inherits from
`ModelWrapper` and does not attempt to auto-wrap it.

## Personalised Layers (FedRep)

If you need backbone/head splitting for FedRep or FedBN:

```python
from quinkgl.models import PyTorchPersonalizedModel
from quinkgl.models.base import ModelSplit

wrapper = PyTorchPersonalizedModel(
    my_model,
    model_split=ModelSplit.auto_detect(
        layer_names=list(my_model.state_dict().keys()),
        num_head_layers=2,
    ),
)
```

Only backbone weights are gossiped; head weights stay strictly local.

## See Also

- [Tutorial T4](../tutorials/T4/index.md) — PyTorch peer scripts
- [API Reference: ModelWrapper](../reference/api/index.md)
