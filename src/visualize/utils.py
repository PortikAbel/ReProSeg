import torch
import torchvision.transforms.functional as F
from PIL import Image
from PIL import ImageDraw as D


def activations_to_alpha(activations: torch.Tensor) -> torch.Tensor:
    """
    Convert activation scores to alpha values for visualization.

    Args:
        activations (torch.Tensor): Activation scores.
        min_activation_score (float): Minimum activation score to consider.

    Returns:
        torch.Tensor: Alpha values for visualization.
    """
    alpha = activations.clone()
    alpha[alpha < alpha.mean()] = 0
    alpha /= alpha.max() if alpha.max() > 0 else 1
    return alpha.unsqueeze(0)  # Add batch dimension for consistency


def prototype_text(p: int, shape: tuple) -> torch.Tensor:
    """
    Generate a tensor representing the prototype text.

    Returns:
        torch.Tensor: A tensor with the prototype text.
    """
    txt_image = Image.new("RGBA", shape, (0, 0, 0))
    draw = D.Draw(txt_image)
    draw.text(
        tuple(s // 2 for s in shape),
        f"Prototype {p}",
        anchor="mm",
        fill="white",
    )
    return F.to_tensor(txt_image)


def draw_activation_minmax_text_on_image(image: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
    """
    Draws min and max activation values as text on the image,
    aligned in the opposite quadrant of the max activation location.
    Args:
        image (torch.Tensor): Image tensor (C, H, W), values in [0,1] or [0,255].
        activations (torch.Tensor): Activation tensor (H, W) or (1, H, W).
    Returns:
        torch.Tensor: Image tensor with text overlay.
    """
    h, w = activations.shape
    min_val = activations.min()
    max_val = activations.max()

    max_idx = torch.argmax(activations)
    row_top = max_idx // w > h // 2
    col_left = max_idx % w > w // 2

    # Calculate the opposite quadrant for text placement
    # Opposite quadrant
    position = (
        10 if col_left else w - 10,
        10 if row_top else h - 10,
    )

    anchor = f"{'l' if col_left else 'r'}{'a' if row_top else 'd'}"

    # Prepare text
    text = f"min: {min_val:.3f}\nmax: {max_val:.3f}"

    pil_img = F.to_pil_image(image)
    draw = D.Draw(pil_img)
    bbox = draw.textbbox(position, text, anchor=anchor)
    draw.rectangle(bbox, fill=(0, 0, 0, 255))
    draw.text(position, text, fill="white", anchor=anchor)

    return F.to_tensor(pil_img)
