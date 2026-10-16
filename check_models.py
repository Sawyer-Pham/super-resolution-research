import torch
from models import SRCNN, ESPCN, EDSR

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using:", device)

# One RGB image: batch size 1, three channels, 32 × 32 pixels.
low_resolution = torch.rand(1, 3, 32, 32).to(device)
target = torch.rand(1, 3, 64, 64).to(device)

for model_class in (SRCNN, ESPCN, EDSR):
    model = model_class(scale=2).to(device)
    output = model(low_resolution)

    assert output.shape == target.shape
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    print(model_class.__name__, output.shape, "OK")