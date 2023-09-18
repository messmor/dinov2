import cv2
import torch
import urllib
import mmcv
from mmcv.runner import load_checkpoint
import urllib
from PIL import Image
from pathlib import Path
from depth_maps_utils import create_depther, render_depth, make_depth_transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')


# Load Model Backbone
BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")
backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
backbone_model.eval()
backbone_model.cuda()

# Load pretrained depth head

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


HEAD_DATASET = "nyu" # in ("nyu", "kitti")
HEAD_TYPE = "dpt" # in ("linear", "linear4", "dpt")


DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

cfg_str = load_config_from_url(head_config_url)
cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

model = create_depther(
    cfg,
    backbone_model=backbone_model,
    backbone_size=BACKBONE_SIZE,
    head_type=HEAD_TYPE,
)

load_checkpoint(model, head_checkpoint_url, map_location="cpu")
model.eval()
model.cuda()




# Show Example Image
image_folder = Path("/home/inseer/data/Hand_Testing/Orientation/Front_Flexion_And_Extension/hand_crops")
save_dir = Path("/home/inseer/data/Hand_Testing/Orientation/Front_Flexion_And_Extension/hand_depth_maps/")
save_dir.mkdir(exist_ok=True, parents=True)
for image_path in image_folder.iterdir():
    image = Image.open(image_path.as_posix())
    # Create Depth Map Estimate
    transform = make_depth_transform()
    scale_factor = 1
    rescaled_image = image.resize((scale_factor * image.width, scale_factor * image.height))
    transformed_image = transform(rescaled_image)
    batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image

    with torch.inference_mode():
        result = model.whole_inference(batch, img_meta=None, rescale=True)

    depth_image = render_depth(result.squeeze().cpu())
    depth_image = np.asarray(depth_image)
    save_path = save_dir / image_path.name
    cv2.imwrite(save_path.as_posix(), depth_image)



