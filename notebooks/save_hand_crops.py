import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

mpl.use('TkAgg')



def convert_pixel_coords_2_numpy(pixel_coords_path):
    """converts pixel coords 2 numpy array with shape (persons, frames, joints, 2)"""
    with open(pixel_coords_path, "r") as file:
        pc = json.load(file)
    np_data = []
    for person_id in pc.keys():
        person_data = []
        for frame in pc[person_id].keys():
            person_data.append(pc[person_id][frame])
        np_data.append(person_data)

    np_data = np.asarray(np_data)

    return np_data[..., 0:2]


def crop_hand(img, box_center, square_size=None):
    """
    crops a square from the image with center and length specified at input
    Args:
        img: numpy array color image
        box_center: (x, y) coordinate of box center
        square_size: float describes square side length of crop
        scale: float
    Returns:
        cropped_image, list of [x_min, y_min, x_max, x_min] used for cropping
    """

    box_center = np.round(np.asarray(box_center)).astype(dtype=int)
    box_x, box_y = box_center

    if box_y >= img.shape[1]:
        box_y = min(box_y, img.shape[1])

    if box_x >= img.shape[0]:
        box_x = min(box_x, img.shape[0])

    # create box corners
    x_min = max(box_x-square_size, 0)
    x_max = min(box_x+square_size, img.shape[0])
    y_min = max(box_y-square_size, 0)
    y_max = min(box_y+square_size, img.shape[1])

    # round to integers
    x_min = np.round(x_min).astype(dtype=int)
    x_max = np.round(x_max).astype(dtype=int)
    y_min = np.round(y_min).astype(dtype=int)
    y_max = np.round(y_max).astype(dtype=int)

    if x_min == x_max:
        return np.zeros((75, 75, 3)), [x_min, x_max, y_min, y_max]

    if y_min == y_max:
        return np.zeros((75, 75, 3)), [x_min, x_max, y_min, y_max]

    return img[x_min:x_max, y_min:y_max], [x_min, y_min, x_max, y_max]


def crop_hands_from_image(image, pixel_coords, side="right"):
    """
    Crops out a subimage around each hand. Returns subimages
    Args:
        image: numpy array of shape (H, W, 3)
        pixel_coords: a numpy array of shape (person_id, joints, 3)
    Returns:
        hands_images: a numpy of shape (person_id, h, w, 3)
    """
    num_persons, num_joints, _ = pixel_coords.shape
    wrist = pixel_coords[:, 10] if side == "right" else pixel_coords[:, 9]
    thumb = pixel_coords[:, 25] if side == "right" else pixel_coords[:, 23]
    knuckle = pixel_coords[:, 26] if side == "right" else pixel_coords[:, 24]
    r_shoulder, l_shoulder = pixel_coords[:, 6], pixel_coords[:, 5]
    collar_len = np.linalg.norm(r_shoulder - l_shoulder, axis=-1)
    palm_len = np.linalg.norm(knuckle - wrist, axis=-1)

    hand_images = []
    for person_id in range(num_persons):
        square_size = max(collar_len[person_id] / 2, 4*palm_len[person_id])
        crop_image, crop_data = crop_hand(image.copy(), box_center=knuckle[person_id], square_size=square_size)
        hand_images.append(crop_image)

    return hand_images


def save_hand_images(pixel_coord_path, video_path, save_dir, downsample=15, plot=False, side="right"):
    pixel_coords = convert_pixel_coords_2_numpy(pixel_coord_path)
    assert Path(video_path).is_file()
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    num_persons = pixel_coords.shape[0]
    frame_num = 0

    while ret:
        if frame_num % downsample != 0:
            frame_num += 1
            ret, frame = video.read()
            continue

        kps = pixel_coords[:, frame_num]
        hand_images = crop_hands_from_image(frame, kps, side=side)
        # plot for testing
        if plot:
            fig, ax = plt.subplots(nrows=num_persons+1)
            ax[0].imshow(frame)
            for i in range(num_persons):
                ax[i+1].imshow(hand_images[i])
            plt.show()
        if save_dir:
            save_img_folder = Path(save_dir) / "hand_crops"
            save_img_folder.mkdir(exist_ok=True, parents=True)
            for i in range(num_persons):
                sp = save_img_folder / f"frame_{frame_num}_person_{i}_{side}.jpg"
                cv2.imwrite(sp.as_posix(), hand_images[i])


        ret, frame = video.read()
        frame_num += 1

    video.release()


if __name__ == "__main__":
    video_name = "Front_Flexion_And_Extension"
    video_folder = Path("/home/inseer/data/Hand_Testing/Orientation/") / video_name
    pc_path = (video_folder / "pixel_coords.json").as_posix()
    video_path = (video_folder / "codec_change_video_blur_none.mp4").as_posix()

    save_hand_images(pc_path, video_path, save_dir=video_folder)



