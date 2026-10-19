"""tacos-predict: predicted instability map of one image from a trained checkpoint."""
import argparse
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from tacos.model import IMAGE_MEAN, IMAGE_STD, TacosNet


def load_model(checkpoint, device="cpu") -> TacosNet:
    """Rebuild a `TacosNet` from the hyperparameters stored in its checkpoint (context statistics included)."""
    return TacosNet.load_from_checkpoint(checkpoint, map_location=device).to(device).eval()


def load_image(path, height=512):
    """(3, H', W') ImageNet-normalized tensor, resized so the height is `height` (the evaluation transform)."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image {path}")
    transform = A.Compose([A.SmallestMaxSize(max_size=height, p=1), A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD, p=1),
                           ToTensorV2()])
    return transform(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))["image"].float()


@torch.no_grad()
def predict_log_map(model, image, context=None):
    """Predicted log-instability map (H, W) of one normalized image (3, H, W)."""
    device = next(model.parameters()).device
    context = None if context is None else torch.as_tensor(context, dtype=torch.float32, device=device)[None]
    return model(image[None].to(device), context)[0, 0].float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("image", help="A labels/images frame (bottom half of the camera image).")
    parser.add_argument("--context", help="Comma-separated context values, in the model's field order "
                                          "(roll,pitch,history_front,history_hind for --context all).")
    parser.add_argument("--height", type=int, default=512, help="Network input height.")
    parser.add_argument("--out", default="cost.png", help="Output png; the map is also saved as a .npy next to it.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    context = [float(v) for v in args.context.split(",")] if args.context else None
    if model.fields and (context is None or len(context) != len(model.fields)):
        parser.error(f"this model needs --context with {len(model.fields)} values: {','.join(model.fields)}")
    cost = np.exp(predict_log_map(model, load_image(args.image, args.height), context))
    out = Path(args.out)
    np.save(out.with_suffix(".npy"), cost)
    scaled = ((cost - cost.min()) / max(float(np.ptp(cost)), 1e-12) * 255).astype(np.uint8)
    cv2.imwrite(str(out), cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO))
    print(f"min {cost.min():.4g}  median {np.median(cost):.4g}  max {cost.max():.4g}  -> {out}")


if __name__ == "__main__":
    main()
