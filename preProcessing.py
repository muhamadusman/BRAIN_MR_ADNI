
import os
import nibabel as nib
import numpy as np
from nilearn.image import resample_img
from scipy.ndimage import center_of_mass

def center_and_zoom(img, target_shape=(256, 256, 256)):
    data = img.get_fdata(dtype=np.float32)
    com = center_of_mass(data > 0)
    com = np.array(com).astype(int)
    
    target_shape = np.array(target_shape, dtype=int)
    start_idx = np.maximum(com - target_shape // 2, 0)
    end_idx = start_idx + target_shape
    
    # Clip to stay within bounds
    for i in range(3):
        if end_idx[i] > data.shape[i]:
            end_idx[i] = data.shape[i]
            start_idx[i] = max(0, end_idx[i] - target_shape[i])
    
    cropped_data = data[
        start_idx[0]:end_idx[0],
        start_idx[1]:end_idx[1],
        start_idx[2]:end_idx[2]
    ]
    
    # Pad if needed
    final_data = np.zeros(target_shape, dtype=cropped_data.dtype)
    pad_start = ((target_shape - np.array(cropped_data.shape)) // 2).astype(int)
    pad_end = pad_start + np.array(cropped_data.shape)
    final_data[
        pad_start[0]:pad_end[0],
        pad_start[1]:pad_end[1],
        pad_start[2]:pad_end[2]
    ] = cropped_data

    # Create new NIfTI (with same affine for now)
    return nib.Nifti1Image(final_data, img.affine, img.header)

def normalize_intensity(img):
    data = img.get_fdata(dtype=np.float32)
    mn, mx = data.min(), data.max()
    if mx > mn:
        data = (data - mn) / (mx - mn)
    else:
        data[:] = 0
    return nib.Nifti1Image(data, img.affine, img.header)

def make_isotropic_affine_preserve_orientation(original_affine, new_spacing=(1,1,1), min_norm=1e-5):
    R = original_affine[:3, :3].copy()
    t = original_affine[:3, 3].copy()
    norms = np.linalg.norm(R, axis=0)  # length of each col

    R_new = np.zeros((3,3), dtype=float)
    for i in range(3):
        if norms[i] < min_norm:
            # degenerate, fallback to canonical axis
            axis = np.zeros(3, dtype=float)
            axis[i] = 1
            R_new[:, i] = axis * new_spacing[i]
        else:
            R_new[:, i] = (R[:, i] / norms[i]) * new_spacing[i]

    new_affine = np.eye(4, dtype=float)
    new_affine[:3, :3] = R_new
    new_affine[:3, 3]  = t
    return new_affine

# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------
input_folder = "/proj/synthetic_alzheimer/users/x_muhak/Data/ADNI_NIFTY_Skullstripped"
output_folder = "/proj/synthetic_alzheimer/users/x_muhak/Data/ADNI_NIFTY_Skullstripped_processed"
os.makedirs(output_folder, exist_ok=True)

for fname in os.listdir(input_folder):
    if fname.endswith(".nii.gz"):
        fpath = os.path.join(input_folder, fname)
        
        # 1) Load
        orig_img = nib.load(fpath)

        # 2) Center to 256^3
        centered_img = center_and_zoom(orig_img, (256, 256, 256))

        # 3) Normalize intensity
        norm_img = normalize_intensity(centered_img)

        # 4) Resample to 1 mm iso, preserving orientation if possible
        try:
            target_affine = make_isotropic_affine_preserve_orientation(norm_img.affine, (1,1,1))
            # If columns are degenerate, they get replaced by standard axes automatically

            # Actually resample to the new shape 256^3
            final_img = resample_img(
                norm_img,
                target_affine=target_affine,
                target_shape=(256, 256, 256),
                interpolation='linear'
            )
        except Exception as e:
            print(f"Warning: could not preserve orientation for {fname} due to {e}")
            # fallback: purely identity approach
            fallback_affine = np.eye(4)
            fallback_affine[:3, :3] = np.diag([1,1,1])
            final_img = resample_img(
                    norm_img,
                    target_affine=target_affine,
                    target_shape=(256, 256, 256),
                    interpolation='linear',
                    copy_header=True,
                    force_resample=True  # or False, depending on your preference
                )


        # 5) Save
        outpath = os.path.join(output_folder, fname)
        nib.save(final_img, outpath)

        # Print final check
        final_zooms = final_img.header.get_zooms()[:3]
        print(f"Processed {fname}: final voxel spacing = {final_zooms}, shape = {final_img.shape}")



# conda create -n pytorch python=3.9
# conda activate pytorch
# conda install numpy scipy nibabel nilearn pytorch torchvision torchaudio cudatoolkit
