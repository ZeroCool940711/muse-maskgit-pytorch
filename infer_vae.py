import torch
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from datasets import load_dataset, Dataset, Image
import os, random, hashlib
from datetime import datetime
from muse_maskgit_pytorch import (
    VQGanVAE,
    VQGanVAETaming,
    get_accelerator,
)
from muse_maskgit_pytorch.dataset import (
    get_dataset_from_dataroot,
    ImageDataset,
)
from tqdm import tqdm
import argparse
import PIL


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_center_crop",
        action="store_true",
        help="Don't do center crop.",
    )
    parser.add_argument(
        "--random_crop",
        action="store_true",
        help="Crop the images at random locations instead of cropping from the center.",
    )
    parser.add_argument(
        "--no_flip",
        action="store_true",
        help="Don't flip image.",
    )
    parser.add_argument(
        "--random_image",
        action="store_true",
        help="Get a random image from the dataset to use for the reconstruction.",
    )
    parser.add_argument(
        "--dataset_save_path",
        type=str,
        default="dataset",
        help="Path to save the dataset if you are making one from a directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility. If set to -1 a random seed will be generated.",
    )
    parser.add_argument(
        "--valid_frac", type=float, default=0.05, help="validation fraction."
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Precision to train on.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to save the training samples and checkpoints",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="results/logs",
        help="Path to log the losses and LR",
    )

    # vae_trainer args
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path to the vae model. eg. 'results/vae.steps.pt'",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the huggingface dataset used.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Dataset folder where your input images for training are.",
    )
    parser.add_argument("--dim", type=int, default=128, help="Model dimension.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch Size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate.")
    parser.add_argument("--vq_codebook_size", type=int, default=256, help="Image Size.")
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size. You may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="This is used to split big images into smaller chunks so we can still reconstruct them no matter the size.",
    )
    parser.add_argument(
        "--min_chunk_size",
        type=int,
        default=8,
        help="We use a minimum chunk size to ensure that the image is always reconstructed correctly.",
    )
    parser.add_argument(
        "--overlap_size",
        type=int,
        default=256,
        help="The overlap size used with --chunk_size to overlap the chunks and make sure the whole image is reconstructe as well as make sure we remove artifacts caused by doing the reconstrucion in chunks.",
    )
    parser.add_argument(
        "--min_overlap_size",
        type=int,
        default=1,
        help="We use a minimum overlap size to ensure that the image is always reconstructed correctly.",
    )
    parser.add_argument(
        "--taming_model_path",
        type=str,
        default=None,
        help="path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)",
    )

    parser.add_argument(
        "--taming_config_path",
        type=str,
        default=None,
        help="path to your trained VQGAN config. This should be a .yaml file. (only valid when taming option is enabled)",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Path to an image to use as input for reconstruction instead of using one from the dataset.",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default=None,
        help="Path to a folder with images to use as input for creating a dataset for reconstructing all the imgaes in it instead of just one image.",
    )
    parser.add_argument(
        "--exclude_folders",
        type=str,
        default=None,
        help="List of folders we want to exclude when doing reconstructions from an input folder.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU to use in case we want to use a specific GPU for inference.",
    )

    # Parse the argument
    return parser.parse_args()


def seed_to_int(s):
    if type(s) is int:
        return s
    if s is None or s == "":
        return random.randint(0, 2**32 - 1)

    if "," in s:
        s = s.split(",")

    if type(s) is list:
        seed_list = []
        for seed in s:
            if seed is None or seed == "":
                seed_list.append(random.randint(0, 2**32 - 1))
            else:
                seed_list = s

        return seed_list

    n = abs(int(s) if s.isdigit() else random.Random(s).randint(0, 2**32 - 1))
    while n >= 2**32:
        n = n >> 32
    return n


def main():
    args = parse_args()
    accelerator = get_accelerator(
        mixed_precision=args.mixed_precision,
        logging_dir=args.logging_dir,
    )

    # set pytorch seed for reproducibility
    torch.manual_seed(seed_to_int(args.seed))

    if args.train_data_dir and not args.input_image and not args.input_folder:
        dataset = get_dataset_from_dataroot(
            args.train_data_dir,
            image_column=args.image_column,
            save_path=args.dataset_save_path,
        )
    elif args.dataset_name and not args.input_image and not args.input_folder:
        dataset = load_dataset(args.dataset_name)["train"]

    elif args.input_image and not args.input_folder:
        # Create dataset from single input image
        dataset = Dataset.from_dict({"image": [args.input_image]}).cast_column("image", Image())

    if args.input_folder:
        # Create dataset from input folder
        extensions = ["jpg", "jpeg", "png", "webp"]
        exclude_folders = args.exclude_folders.split(',') if args.exclude_folders else []

        filepaths = []
        for root, dirs, files in os.walk(args.input_folder, followlinks=True):
            # Resolve symbolic link to actual path and exclude based on actual path
            resolved_root = os.path.realpath(root)
            for exclude_folder in exclude_folders:
                if exclude_folder in resolved_root:
                    dirs[:] = []
                    break
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    filepaths.append(os.path.join(root, file))

        if not filepaths:
            print(f"No images with extensions {extensions} found in {args.input_folder}.")
            sys.exit(1)

        dataset = Dataset.from_dict({"image": filepaths}).cast_column("image", Image())




    if args.vae_path and args.taming_model_path:
        raise Exception("You can't pass vae_path and taming args at the same time.")

    if args.vae_path:
        accelerator.print("Loading Muse VQGanVAE")
        vae = VQGanVAE(dim=args.dim, vq_codebook_size=args.vq_codebook_size).to(
            accelerator.device if args.gpu == 0 else f"cuda:{args.gpu}"
        )

        accelerator.print("Resuming VAE from: ", args.vae_path)
        vae.load(
            args.vae_path
        )  # you will want to load the exponentially moving averaged VAE

    elif args.taming_model_path:
        print("Loading Taming VQGanVAE")
        vae = VQGanVAETaming(
            vqgan_model_path=args.taming_model_path,
            vqgan_config_path=args.taming_config_path,
        )
        args.num_tokens = vae.codebook_size
        args.seq_len = vae.get_encoded_fmap_size(args.image_size) ** 2
    vae = vae.to(accelerator.device if args.gpu == 0 else f"cuda:{args.gpu}")
    # then you plug the vae and transformer into your MaskGit as so

    dataset = ImageDataset(
        dataset,
        args.image_size,
        image_column=args.image_column,
        center_crop=True if not args.no_center_crop and not args.random_crop else False,
        flip=not args.no_flip,
        random_crop=args.random_crop if args.random_crop else False
    )

    if args.input_image and not args.input_folder:
        image_id = 0 if not args.random_image else random.randint(0, len(dataset))

        os.makedirs(f"{args.results_dir}/outputs", exist_ok=True)

        save_image(dataset[image_id], f"{args.results_dir}/outputs/input.{str(args.input_image).split('.')[-1]}")

        _, ids, _ = vae.encode(dataset[image_id][None].to(accelerator.device if args.gpu == 0 else f"cuda:{args.gpu}"))
        recon = vae.decode_from_ids(ids)
        save_image(recon, f"{args.results_dir}/outputs/output.{str(args.input_image).split('.')[-1]}")


    if args.input_folder:
        # Create output directory and save input images and reconstructions as grids
        output_dir = os.path.join(args.results_dir, "outputs", os.path.basename(args.input_folder))
        os.makedirs(output_dir, exist_ok=True)
        for i in tqdm(range(len(dataset))):
            # Get single image tensor from batch
            input_image = dataset[i]
            input_image = input_image.unsqueeze(0)  # add a batch dimension

            # Set starting chunk size and overlap size
            chunk_size = args.chunk_size
            overlap_size = args.overlap_size

            # Try encoding and decoding with increasing smaller chunk and overlap sizes until success
            while True:
                try:
                    _, ids, _ = vae.encode(input_image.to(accelerator.device if args.gpu == 0 else f"cuda:{args.gpu}"))
                    recon = vae.decode_from_ids(ids)
                    break  # If we made it this far, we didn't run out of memory
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Try again with smaller chunk and overlap sizes
                        print("Out of memory error encountered. Trying with smaller chunk and overlap sizes.")
                        chunk_size //= 2
                        overlap_size //= 2

                        # Keep dividing chunk and overlap sizes until success or we reach minimum size
                        while chunk_size >= args.min_chunk_size and overlap_size >= args.min_overlap_size:
                            try:
                                _, ids, _ = vae.encode(input_image.to(accelerator.device if args.gpu == 0 else f"cuda:{args.gpu}"))
                                recon = vae.decode_from_ids(ids)
                                break  # If we made it this far, we didn't run out of memory
                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    # Try again with smaller chunk and overlap sizes
                                    print("Out of memory error encountered. Trying with smaller chunk and overlap sizes.")
                                    chunk_size //= 2
                                    overlap_size //= 2
                                else:
                                    # Some other kind of RuntimeError occurred, so re-raise it
                                    raise e

                        # If we've reached minimum size, raise a RuntimeError
                        if chunk_size < args.min_chunk_size or overlap_size < args.min_overlap_size:
                            print ("Out of memory even with the smallest chunk and overlap sizes.")
                            print (f"Skipping image.")
                            pass
                    else:
                        # Some other kind of RuntimeError occurred, so re-raise it
                        raise e

            # Convert input_image and recon to PIL images
            input_image = F.to_pil_image(input_image.squeeze(0).cpu(), mode="RGB")
            recon = F.to_pil_image(recon.squeeze(0).cpu(), mode="RGB")

            # Convert color space of recon to match that of input_image
            #recon = recon.convert(input_image.mode)

            # Combine input_image and recon into a grid
            grid_image = PIL.Image.new('RGB', (input_image.width * 2, input_image.height))
            grid_image.paste(input_image, (0, 0))
            grid_image.paste(recon, (input_image.width, 0))


            # Save grid
            now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            hash = hashlib.sha1(input_image.tobytes()).hexdigest()
            try:
                filename = f"{hash}_{now}.{str(args.input_image).split('.')[-1]}"
                grid_image.save(os.path.join(output_dir, filename))
            except ValueError:
                filename = f"{hash}_{now}.png"
                grid_image.save(os.path.join(output_dir, filename))

if __name__ == "__main__":
    main()
