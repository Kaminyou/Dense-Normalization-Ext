import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.model import get_model
from utils.dataset import XInferenceDataset, XPrefetchInferenceDataset
from utils.util import (
    read_yaml_config,
    reverse_image_normalize,
    test_transforms,
)


def main():
    parser = argparse.ArgumentParser("Model inference")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./data/example/config.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    config = read_yaml_config(args.config)

    model = get_model(
        config=config,
        model_name=config["MODEL_NAME"],
        normalization=config["INFERENCE_SETTING"]["NORMALIZATION"],
        isTrain=False,
        parallelism=config["INFERENCE_SETTING"].get('PARALLELISM', False),
    )

    if config["INFERENCE_SETTING"]["NORMALIZATION"] == "tin":
        test_dataset = XInferenceDataset(
            root_X=config["INFERENCE_SETTING"]["TEST_DIR_X"],
            transform=test_transforms,
            return_anchor=True,
            thumbnail=config["INFERENCE_SETTING"]["THUMBNAIL"],
        )
    elif config["INFERENCE_SETTING"]["NORMALIZATION"] == "iin" and config["INFERENCE_SETTING"].get('PARALLELISM', False):
        test_dataset = XPrefetchInferenceDataset(
            root_X=config["INFERENCE_SETTING"]["TEST_DIR_X"],
            transform=test_transforms,
            return_anchor=True,
        )
    else:
        test_dataset = XInferenceDataset(
            root_X=config["INFERENCE_SETTING"]["TEST_DIR_X"],
            transform=test_transforms,
            return_anchor=True,
        )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, pin_memory=True,
    )

    model.load_networks(config["INFERENCE_SETTING"]["MODEL_VERSION"])

    basename = os.path.basename(config["INFERENCE_SETTING"]["TEST_X"])
    filename = os.path.splitext(basename)[0]
    save_path_root = os.path.join(
        config["EXPERIMENT_ROOT_PATH"],
        config["EXPERIMENT_NAME"],
        "test",
        filename,
    )

    if (
        "OVERWRITE_OUTPUT_PATH" in config["INFERENCE_SETTING"]
        and config["INFERENCE_SETTING"]["OVERWRITE_OUTPUT_PATH"] != ""
    ):
        save_path_root = config["INFERENCE_SETTING"]["OVERWRITE_OUTPUT_PATH"]

    save_path_base = os.path.join(
        save_path_root,
        config["INFERENCE_SETTING"]["NORMALIZATION"],
        config["INFERENCE_SETTING"]["MODEL_VERSION"],
    )
    os.makedirs(save_path_base, exist_ok=True)
    print(save_path_base)

    if config["INFERENCE_SETTING"]["NORMALIZATION"] == "tin":
        total_time = 0
        cnt = 0
        model.init_thumbnail_instance_norm_for_whole_model()
        thumbnail = test_dataset.get_thumbnail()
        now = time.time()
        thumbnail_fake = model.inference(thumbnail)
        total_time += time.time() - now
        save_image(
            reverse_image_normalize(thumbnail_fake),
            os.path.join(save_path_base, "thumbnail_Y_fake.png"),
        )

        model.use_thumbnail_instance_norm_for_whole_model()
        for idx, data in enumerate(test_loader):
            print(f"Processing {idx}", end="\r")
            X, X_path = data["X_img"], data["X_path"]
            now = time.time()
            cnt += 1
            Y_fake = model.inference(X)
            total_time += time.time() - now
            if config["INFERENCE_SETTING"]["SAVE_ORIGINAL_IMAGE"]:
                save_image(
                    reverse_image_normalize(X),
                    os.path.join(
                        save_path_base,
                        f"{Path(X_path[0]).stem}_X_{idx}.png",
                    ),
                )
            save_image(
                reverse_image_normalize(Y_fake),
                os.path.join(
                    save_path_base, f"{Path(X_path[0]).stem}_Y_fake_{idx}.png"
                ),
            )
        print('time: ', total_time, cnt)

    elif config["INFERENCE_SETTING"]["NORMALIZATION"] == "kin":
        save_path_base_kin = os.path.join(
            save_path_base,
            f"{config['INFERENCE_SETTING']['KIN_KERNEL']}_"
            f"{config['INFERENCE_SETTING']['KIN_PADDING']}",
        )
        os.makedirs(save_path_base_kin, exist_ok=True)
        y_anchor_num, x_anchor_num = test_dataset.get_boundary()
        # as the anchor num from 0 to N,
        # anchor_num = N but it actually has N + 1 values
        model.init_kernelized_instance_norm_for_whole_model(
            y_anchor_num=y_anchor_num + 1,
            x_anchor_num=x_anchor_num + 1,
            kernel_padding=config["INFERENCE_SETTING"]["KIN_PADDING"],
            kernel_mode=config["INFERENCE_SETTING"]["KIN_KERNEL"],
        )
        total_time = 0
        cnt = 0
        for idx, data in enumerate(test_loader):
            print(f"Caching {idx}", end="\r")
            X, X_path, y_anchor, x_anchor = (
                data["X_img"],
                data["X_path"],
                data["y_idx"],
                data["x_idx"],
            )
            now = time.time()
            cnt += 1
            _ = model.inference_with_anchor(
                X,
                y_anchor=y_anchor,
                x_anchor=x_anchor,
                padding=config["INFERENCE_SETTING"]["KIN_PADDING"],
            )
            total_time += time.time() - now
        now = time.time()
        model.use_kernelized_instance_norm_for_whole_model(
            padding=config["INFERENCE_SETTING"]["KIN_PADDING"]
        )
        total_time += time.time() - now
        for idx, data in enumerate(test_loader):
            print(f"Processing {idx}", end="\r")
            X, X_path, y_anchor, x_anchor = (
                data["X_img"],
                data["X_path"],
                data["y_idx"],
                data["x_idx"],
            )
            now = time.time()
            Y_fake = model.inference_with_anchor(
                X,
                y_anchor=y_anchor,
                x_anchor=x_anchor,
                padding=config["INFERENCE_SETTING"]["KIN_PADDING"],
            )
            total_time += time.time() - now
            if config["INFERENCE_SETTING"]["SAVE_ORIGINAL_IMAGE"]:
                save_image(
                    reverse_image_normalize(X),
                    os.path.join(
                        save_path_base_kin,
                        f"{Path(X_path[0]).stem}_X_{idx}.png",
                    ),
                )
            save_image(
                reverse_image_normalize(Y_fake),
                os.path.join(
                    save_path_base_kin,
                    f"{Path(X_path[0]).stem}_Y_fake_{idx}.png",
                ),
            )
        print('time:', total_time, cnt)
    
    elif config["INFERENCE_SETTING"]["NORMALIZATION"] == "iin" and not config["INFERENCE_SETTING"].get('PARALLELISM', False):
        os.makedirs(save_path_base, exist_ok=True)
        y_anchor_num, x_anchor_num = test_dataset.get_boundary()
        print(y_anchor_num, x_anchor_num)
        # as the anchor num from 0 to N,
        # anchor_num = N but it actually has N + 1 values
        model.init_interpolated_instance_norm_for_whole_model(
            y_anchor_num=y_anchor_num + 1,
            x_anchor_num=x_anchor_num + 1,
        )
        total_time = 0
        cnt = 0
        for idx, data in enumerate(test_loader):
            print(f"Caching {idx}", end="\r")
            X, X_path, y_anchor, x_anchor = (
                data["X_img"],
                data["X_path"],
                data["y_idx"],
                data["x_idx"],
            )
            now = time.time()
            cnt += 1
            _ = model.inference_with_anchor(
                X,
                y_anchor=y_anchor,
                x_anchor=x_anchor,
                padding=1,
            )
            total_time += time.time() - now

        model.use_interpolated_instance_norm_for_whole_model()
        for idx, data in enumerate(test_loader):
            print(f"Processing {idx}", end="\r")
            X, X_path, y_anchor, x_anchor = (
                data["X_img"],
                data["X_path"],
                data["y_idx"],
                data["x_idx"],
            )
            now = time.time()
            Y_fake = model.inference_with_anchor(
                X,
                y_anchor=y_anchor,
                x_anchor=x_anchor,
                padding=1,
            )
            total_time += time.time() - now
            if config["INFERENCE_SETTING"]["SAVE_ORIGINAL_IMAGE"]:
                save_image(
                    reverse_image_normalize(X),
                    os.path.join(
                        save_path_base,
                        f"{Path(X_path[0]).stem}_X_{idx}.png",
                    ),
                )
            save_image(
                reverse_image_normalize(Y_fake),
                os.path.join(
                    save_path_base,
                    f"{Path(X_path[0]).stem}_Y_fake_{idx}.png",
                ),
            )
        print('time:', total_time, cnt)

    elif config["INFERENCE_SETTING"]["NORMALIZATION"] == "iin" and config["INFERENCE_SETTING"].get('PARALLELISM', False):
        os.makedirs(save_path_base, exist_ok=True)
        y_anchor_num, x_anchor_num = test_dataset.get_boundary()
        # as the anchor num from 0 to N,
        # anchor_num = N but it actually has N + 1 values
        model.init_prefetch_interpolated_instance_norm_for_whole_model(
            y_anchor_num=y_anchor_num + 1,
            x_anchor_num=x_anchor_num + 1,
        )
        total_time = 0
        cnt = 0
        for idx, data in enumerate(test_loader):
            print(f"Executing {idx}", end="\r")
            X = torch.cat((data['X_img'], data['pre_img']), dim=0)
            now = time.time()
            cnt += 1
            Y_fake = model.inference_with_anchor(
                X,
                y_anchor=int(data['y_idx'][0]),
                x_anchor=int(data['x_idx'][0]),
                padding=1,
                pre_y_anchor=int(data['pre_y_idx'][0]),
                pre_x_anchor=int(data['pre_x_idx'][0]),
            )
            total_time += time.time() - now
            Y_fake = Y_fake[[0]]
            pad = 16
            Y_fake = Y_fake[:, :, pad:512+pad, pad:512+pad]
            if data['y_idx'][0] != -1:
                X_path = data['X_path']
                save_image(
                    reverse_image_normalize(Y_fake),
                    os.path.join(
                        save_path_base,
                        f"{Path(X_path[0]).stem}_Y_fake_{idx}.png",
                    ),
                )
        print('time:', total_time, cnt)

    else:
        total_time = 0
        cnt = 0
        for idx, data in enumerate(test_loader):
            print(f"Processing {idx}", end="\r")

            X, X_path = data["X_img"], data["X_path"]
            now = time.time()
            cnt += 1
            Y_fake = model.inference(X)
            total_time += time.time() - now

            if config["INFERENCE_SETTING"]["SAVE_ORIGINAL_IMAGE"]:
                save_image(
                    reverse_image_normalize(X),
                    os.path.join(
                        save_path_base,
                        f"{Path(X_path[0]).stem}_X_{idx}.png",
                    ),
                )

            save_image(
                reverse_image_normalize(Y_fake),
                os.path.join(
                    save_path_base, f"{Path(X_path[0]).stem}_Y_fake_{idx}.png"
                ),
            )
        print('time:', total_time, cnt)


if __name__ == "__main__":
    main()
