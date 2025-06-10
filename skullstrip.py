import os
import logging
import nibabel as nib
from subprocess import run

# Input and output directories
data_dir = '/flush/muhak80/Lund/raw_ADNI_ImagingData/ADNI_latest'  # Path to raw image directory
output_dir = '/flush/muhak80/Lund/raw_ADNI_ImagingData/ADNI_latest_skullstrip'  # Path to output directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Set up logging
logging.basicConfig(filename="skull_stripping.log", level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# BET Parameters
frac = 0.1  # Lower value for stricter extraction
robust = True  # Use robust mode (-R)
reduce_eyes = True  # Reduce eyes (-S)

# HD-BET Option
use_hd_bet = True

def apply_mask(input_file, mask_file, output_file):
    """
    Apply a binary mask to the image.
    """
    img = nib.load(input_file)
    mask = nib.load(mask_file)
    masked_data = img.get_fdata() * mask.get_fdata()
    masked_img = nib.Nifti1Image(masked_data, img.affine)
    nib.save(masked_img, output_file)
    logging.info(f"Applied mask to {input_file}, saved to {output_file}")

for file in os.listdir(data_dir):
    if not (file.endswith(".nii") or file.endswith(".nii.gz")):
        print(f"Skipping non-NIfTI file: {file}")
        continue

    try:
        input_file = os.path.join(data_dir, file)
        output_file = os.path.join(output_dir, file)
        mask_file = os.path.join(output_dir, file.replace(".nii.gz", "_mask.nii.gz").replace(".nii", "_mask.nii.gz"))

        if os.path.exists(output_file):
            print(f"Output already exists for {file}, skipping.")
            continue

        if use_hd_bet:
            # Use HD-BET
            hd_bet_output_base = os.path.join(output_dir, os.path.splitext(file)[0] + '_bet.nii.gz')  # Base name for HD-BET output
            run(["hd-bet", "-i", input_file, "-o", hd_bet_output_base])  # Removed -mode fast
            skullstripped_file = f"{hd_bet_output_base}.nii.gz"
            hd_bet_mask_file = f"{hd_bet_output_base}_mask.nii.gz"

            # Rename files to correct naming convention
            if os.path.exists(skullstripped_file):
                print(f"Skull-stripped file found: {skullstripped_file}")
                os.rename(skullstripped_file, output_file)
            else:
                logging.warning(f"Skull-stripped file not found for {file}, skipping.")
                continue
            if os.path.exists(hd_bet_mask_file):
                print(f"Mask file found: {hd_bet_mask_file}")
                os.rename(hd_bet_mask_file, mask_file)
            else:
                logging.warning(f"Mask file not found for {file}, skipping.")
                continue

            print(f"{file} processed with HD-BET")
            logging.info(f"Processed {file} with HD-BET")
        else:
            # Use BET
            bet_command = ["bet", input_file, output_file, "-f", str(frac)]
            if robust:
                bet_command.append("-R")
            if reduce_eyes:
                bet_command.append("-S")
            bet_command.append("-m")
            run(bet_command)

            # Ensure proper naming for BET-generated mask
            generated_mask_file = output_file.replace(".nii.gz", "_mask.nii.gz")
            os.rename(generated_mask_file, mask_file)

            # Apply mask
            apply_mask(input_file, mask_file, output_file)
            print(f"{file} processed with BET")
            logging.info(f"Processed {file} with BET")

    except Exception as e:
        print(f"Error processing {file}: {e}")
        logging.error(f"Error processing {file}: {e}")
