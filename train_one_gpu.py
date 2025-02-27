# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
join = os.path.join
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from utils.SurfaceDice import compute_dice_coefficient
from open_clip import create_model_from_pretrained
from get_clip_embedding1 import get_clip_embeddings, ModifiedCLIPModel
from get_clip_embedding1 import create_modified_clip_model
import torchvision.transforms as T
from sklearn.model_selection import train_test_split


import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

# -*- coding: utf-8 -*-
"""
Train the image encoder and mask decoder.
Freeze prompt image encoder.
"""
# Helper functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

# Dataset class for loading .npz files
class NpyDataset(Dataset):
    def __init__(self, data_roots, bbox_shift=20, tokenizer_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        """
        Dataset for loading .npz files of medical images and labels.

        Args:
            data_roots (str): Root directory containing 'imgs' and 'gts' subdirectories.
            bbox_shift (int): Random shift range for bounding box calculation.
            tokenizer_name (str): Huggingface tokenizer name for text descriptions.
        """
        self.data_roots = data_roots
        self.slices = []  # Metadata for slices
        self.bbox_shift = bbox_shift
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        logger.info(f"Initializing dataset from: {data_roots}")

        # Iterate over organ directories
        for organ_name in ["liver", "kidney", "spleen", "pancreas"]:
            logger.info(f"Processing organ: {organ_name}")
            img_dir = os.path.join(data_roots, "imgs", organ_name)
            gt_dir = os.path.join(data_roots, "gts", organ_name)
            description_file = os.path.join(img_dir, "descriptions.txt")

            # Load organ description
            if os.path.isfile(description_file):
                with open(description_file, 'r') as f:
                    description = f.read().strip()
            else:
                description = f"{organ_name} medical image segmentation"

            # Collect all image and ground truth file paths
            organ_img_paths = sorted(glob.glob(os.path.join(img_dir, "*.npz")))
            organ_gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.npz")))

            assert len(organ_img_paths) == len(organ_gt_paths), f"Mismatch between images and labels for {organ_name}"

            # Process each file and add all slices to the dataset
            for img_path, gt_path in zip(organ_img_paths, organ_gt_paths):
                img_data = np.load(img_path, mmap_mode="r")
                gt_data = np.load(gt_path, mmap_mode="r")
                num_slices = img_data["imgs"].shape[0]

                self.slices.extend([
                    (img_path, gt_path, slice_idx, description) for slice_idx in range(num_slices)
                ])

                logger.info(f"Added {num_slices} slices from {img_path}")

        logger.info(f"Dataset initialized. Total slices: {len(self.slices)}")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        """
        Load a single slice, compute bounding box, and tokenize description.

        Args:
            index (int): Index of the data sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor]:
                - Image tensor of shape [C, H, W].
                - Ground truth tensor of shape [1, H, W].
                - Bounding box tensor of shape [4].
                - Image file name.
                - Tokenized text description tensor of shape [seq_len].
        """
        img_path, gt_path, slice_idx, description = self.slices[index]

        # Load image and ground truth slices
        img_npz = np.load(img_path, mmap_mode="r")
        gt_npz = np.load(gt_path, mmap_mode="r")
        img = img_npz["imgs"][slice_idx]  # Specific slice
        gt = gt_npz["gts"][slice_idx]

        # Ensure image has [C, H, W] format
        if len(img.shape) == 3:  # If RGB [H, W, C]
            img = np.transpose(img, (2, 0, 1))  # Convert to [C, H, W]
        else:  # If grayscale [H, W]
            img = img[np.newaxis, :, :]  # Add channel dimension: [1, H, W]

        gt = np.uint8(gt > 0)  # Binary mask

        # Compute bounding box
        y_indices, x_indices = np.where(gt > 0)
        if len(x_indices) > 0:  # Ensure there are labeled regions
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = gt.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])
        else:
            bboxes = np.array([0, 0, 0, 0])  # Default bounding box for empty regions

        # Tokenize the description
        text_tokens = self.tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=77
        )

        return (
            torch.tensor(img).float(),
            torch.tensor(gt).unsqueeze(0).long(),  # Add channel dimension
            torch.tensor(bboxes).float(),
            f"{os.path.basename(img_path)}_slice{slice_idx}",
            text_tokens['input_ids'].squeeze(0),
        )


# MedSAM model
class MedSAM(nn.Module):
    def __init__(self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        use_clip=False,

    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.use_clip = use_clip

        # Freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        
        for name, param in self.image_encoder.named_parameters():
            if "neck_list" in name or "neck" in name:
                param.requires_grad = True

        if self.use_clip:
            clip_model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            self.embed_dim = 256
            num_heads = 8
            self.clip_model, self.clip_preprocess = create_model_from_pretrained(clip_model_name)
            self.modified_clip_model = ModifiedCLIPModel(self.clip_model, self.embed_dim, num_heads)
        else:
            self.modified_clip_model = None
            self.clip_model = None
            self.clip_preprocess = None

    def get_clip_embeddings(self, images, text_inputs):

        if not self.use_clip:
            return None

        with torch.no_grad():
            processed_images = []
            for image in images:
                if isinstance(image, torch.Tensor):
                    image = T.ToPILImage()(image)
                processed_images.append(self.clip_preprocess(image).unsqueeze(0))

            clip_inputs = torch.cat(processed_images, dim=0)

        attn_output = self.modified_clip_model(clip_inputs.to(images.device), text_inputs.to(images.device))
        clip_image_embeddings = attn_output
        clip_prompt_embeddings = clip_image_embeddings.view(
            clip_image_embeddings.size(0),
            1,
            self.embed_dim,
        )
        return clip_prompt_embeddings

    def forward(self, image, box, text_input):

        image_embedding = self.image_encoder(image)
        device = image_embedding[0].device if isinstance(image_embedding, list) else image_embedding.device

        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None
            )

        # !!!!!!
        clip_embeddings = None
        if self.use_clip:
            clip_embeddings = self.get_clip_embeddings(image, text_input).to(device)

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            clip_prompt_embeddings=clip_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

def train_val_split(dataset, val_ratio=0.2, seed=2023, val_organs=["liver", "kidney", "spleen", "pancreas"]):
    """
    Split the dataset into training and validation subsets by assigning entire `.npz` files (patients)
    to either train or validation to ensure no patient-level overlap. Validation subsets are organized by organ.

    Args:
        dataset (NpyDataset): The dataset to split.
        val_ratio (float): The fraction of patients to allocate to the validation set.
        seed (int): Random seed for reproducibility.
        val_organs (list): List of organs to include in the validation set.

    Returns:
        Tuple[Subset, Dict[str, Subset]]: Training dataset and a dictionary of validation datasets (by organ).
    """

    # Extract unique patient file paths
    patient_files = list({img_path for _, img_path, _, _ in dataset.slices})

    # Split patient files into train and validation sets
    train_files, val_files = train_test_split(
        patient_files, test_size=val_ratio, random_state=seed
    )

    # Create indices for train and validation
    train_indices = [
        idx for idx, (_, img_path, _, _) in enumerate(dataset.slices) if img_path in train_files
    ]
    val_indices = [
        idx for idx, (_, img_path, _, _) in enumerate(dataset.slices) if img_path in val_files
    ]

    # Further divide validation set by organ
    val_datasets = {}
    for organ in val_organs:
        organ_val_indices = [
            idx for idx in val_indices if f"/{organ}/" in dataset.slices[idx][1]
        ]
        if organ_val_indices:  # Only include organ subsets with data
            val_datasets[organ] = torch.utils.data.Subset(dataset, organ_val_indices)

    # Create training subset
    train_dataset = torch.utils.data.Subset(dataset, train_indices)

    # Debugging: Print statistics
    print(f"Total patients: {len(patient_files)}")
    print(f"Training patients: {len(train_files)}, Validation patients: {len(val_files)}")
    print(f"Training samples: {len(train_indices)}")
    for organ, subset in val_datasets.items():
        print(f"Validation samples for {organ}: {len(subset)}")

    return train_dataset, val_datasets




# Main training script
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--tr_npy_path", type=str, default="data/Flare2021npz")
    parser = argparse.ArgumentParser()
    parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--load_pretrain", type=bool, default=True, help="Load pretrain model")
    parser.add_argument("-pretrain_model_path", type=str, default="")
    parser.add_argument("-work_dir", type=str, default="./work_dir")
    parser.add_argument("-num_epochs", type=int, default=100)
    parser.add_argument("-batch_size", type=int, default=4)
    parser.add_argument("-num_workers", type=int, default=2)
    parser.add_argument("-weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("-lr", type=float, default=0.0001, metavar="LR", help="Learning rate")
    parser.add_argument("-use_wandb", type=bool, default=False, help="Use wandb for training log")
    parser.add_argument("-use_amp", action="store_true", default=False, help="Use AMP")
    parser.add_argument("--resume", type=str, default="", help="Resume training from checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")

    ### new params
    parser.add_argument("--ms_features", action="store_true")
    parser.add_argument("--one_neck", action="store_true")
    parser.add_argument("--use_clip", type=bool, default=True,help="Whether to use CLIP model for text and image prompt fusion")

    args = parser.parse_args()
    print("Clearing CUDA cache...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("CUDA cache cleared.")
    join = os.path.join
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    use_clip_str = "_use_clip" if args.use_clip else "_no_clip"
    model_save_path = join(args.work_dir,
                           args.task_name + f"_MS{args.ms_features}" + f"_oneneck{args.one_neck}" + use_clip_str + "_" + run_id)

    #args = parser.parse_args()
    device = torch.device(args.device)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        print(f"Created directory: {model_save_path}")
    
    ### ======== modify !!!!!!!!!!!!!!!!!!!! ======== ###
    sam_model = sam_model_registry[args.model_type](
        checkpoint=args.checkpoint,
        ms_features=args.ms_features,
        one_neck=args.one_neck,
    )
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(args.device)
    medsam_model.train()

    optimizer = torch.optim.AdamW(
        medsam_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True)
    ce_loss = nn.BCEWithLogitsLoss()
    full_dataset = NpyDataset("data/Flare2021npz")
    train_dataset, val_datasets = train_val_split(full_dataset)

    print(f"Number of training samples: {len(train_dataset)}")
    for organ, val_dataset in val_datasets.items():
        print(f"Validation set for {organ}: {len(val_dataset)}")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Ensure 'state_dict' is extracted if the checkpoint contains additional metadata
    if 'model' in checkpoint:
        state_dict = checkpoint['model']  # Adjust this key based on the actual checkpoint structure
    else:
        state_dict = checkpoint

# Load the extracted state_dict into the model
    sam_model.load_state_dict(state_dict, strict=False)

# Optionally, extract the starting epoch if available
    start_epoch = checkpoint.get('epoch', 0)
    start_epoch+=1

    print(f"Checkpoint loaded successfully. Starting from epoch {start_epoch}.")
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        #Store training and validation metrics
    train_losses = []
    train_accuracy = []
    val_losses = {organ: [] for organ in val_datasets.keys()}
    val_accuracies = {organ: [] for organ in val_datasets.keys()}
    start_epoch =0
    # num_epochs=100
    best_loss = 1e10
    iter_num=0
        
    for epoch in range(start_epoch, args.num_epochs):
        medsam_model.train()
        epoch_loss = 0
        epoch_dice = 0
        for step, (image, gt2D, boxes, img_name, text_input) in enumerate(tqdm(train_dataloader)):
            # break
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                        medsam_pred, gt2D.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred = medsam_model(image, boxes_np, text_input)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            dice_coefficient = compute_dice_coefficient(gt2D.detach().cpu().numpy(), medsam_pred.detach().cpu().numpy() > 0.5 )
            epoch_dice += dice_coefficient

            epoch_loss += loss.item()
            # print(epoch_loss)
            iter_num += 1

        epoch_loss /= len(train_dataloader)
        epoch_dice /= len(train_dataloader)
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_dice)

        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss}, {"epoch_dice": epoch_dice})# Validation loop

        medsam_model.eval()
        val_results = defaultdict(dict)
        with torch.no_grad():
            for organ, val_dataset in val_datasets.items():
                val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,)
                total_loss, total_dice, count = 0, 0, 0

                # use tqdm to show progress bar
                for image, gt, boxes, _, text_input in tqdm(val_loader, desc=f"Validating {organ}", leave=False):
                    bboxes = boxes.numpy()
                    image, gt = image.to(device), gt.to(device)

                    # model prediction
                    medsam_pred = medsam_model(image, bboxes, text_input)
                    loss = seg_loss(medsam_pred, gt) + ce_loss(medsam_pred, gt.float())
                    dice = compute_dice_coefficient(
                        gt.cpu().numpy(), (medsam_pred > 0.5).cpu().numpy()
                    )

                    # total loss and dice
                    total_loss += loss.item()
                    total_dice += dice
                    count += 1

                # compute average loss and dice on validation set
                val_results[organ]["loss"] = total_loss / count
                val_results[organ]["dice"] = total_dice / count
                val_losses[organ].append(val_results[organ]["loss"])
                val_accuracies[organ].append(val_results[organ]["dice"])
                print(f"{organ.capitalize()} Validation Complete: Loss: {val_results[organ]['loss']:.4f}, Dice: {val_results[organ]['dice']:.4f}")
           
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, train_Loss: {epoch_loss}, train_accuracy : {epoch_dice}'
        )
        # Logging metrics
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        ## save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        
        total_val_loss = 0 
        for loss in val_losses.values():
            total_val_loss += loss[-1] / len(val_losses)
        
        ## save the best model
        if total_val_loss < best_loss:
            best_loss = total_val_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))
            print("saved a better ckpt.")

        print("val loss:", total_val_loss)
        for organ, metrics in val_results.items():
            print(f"  {organ.capitalize()} - Val Loss: {metrics['loss']:.4f}, Dice: {metrics['dice']:.4f}")
        
        # %% plot loss
    
        # Update loss plots
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        for organ, losses in val_losses.items():
            plt.plot(losses, label=f"Val Loss ({organ})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracy, label="Train accuracy")
        for organ, accuracies in val_accuracies.items():
            plt.plot(accuracies, label=f"Val Dice ({organ})")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.title("Validation Dice Score")
        plt.legend()

        plt.tight_layout()
        plt.savefig(join(model_save_path, "training_curves.png"))
        plt.close()

if __name__ == "__main__":
    main()
