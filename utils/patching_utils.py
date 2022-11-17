import glob
import os

import numpy as np
from PIL import Image
from empatches import EMPatches


def patchify(img_dir, dump_dir, patch_size=512, overlap=0.5):
    os.makedirs(dump_dir, exist_ok=True)
    emp = EMPatches()

    img_files = sorted(glob.glob(img_dir + "/*"))

    for img_path in img_files:
        img = np.array(Image.open(img_path))

        img_patches, img_indices = emp.extract_patches(img, patchsize=patch_size, overlap=overlap)

        for patch, index in zip(img_patches, img_indices):
            img_fname = img_path.split('/')[-1]
            Image.fromarray(patch).save(f"{dump_dir}/{img_fname.split('.')[0]}_{index}.{img_fname.split('.')[-1]}")


def assemble(img_dir):
    merged_imgs = {}
    emp = EMPatches()

    img_names = set([elem.split("/")[-1].split("_(")[0] for elem in sorted(glob.glob(img_dir + "/*"))])

    for img_name in img_names:
        imgs, indc = [], []
        for img_path in sorted(glob.glob(img_dir + "/*")):
            if img_name in img_path:
                imgs.append(np.array(Image.open(img_path)))
                indc.append(eval("(" + img_path.split("(")[1].split(")")[0] + ")"))

        zipped = zip(indc, imgs)
        res = np.array(sorted(zipped, key=lambda x: x[0]))
        merged_img = emp.merge_patches(res[:, 1], res[:, 0]).astype(np.uint8)

        merged_imgs[img_name] = merged_img
        Image.fromarray(merged_img).save(img_name + ".jpg")
    return merged_imgs


if __name__ == "__main__":
    for subset in ["train", "val", "test"]:
        img_dir = f"../../data/idrid_hard_exudate_segmentation/img/{subset}"
        dump_dir = f"../../data/idrid_hard_exudate_segmentation512/img/{subset}"
        patchify(img_dir, dump_dir)

        grd_dir = f"../../data/idrid_hard_exudate_segmentation/grd/{subset}"
        dump_dir = f"../../data/idrid_hard_exudate_segmentation512/grd/{subset}"
        patchify(grd_dir, dump_dir)
