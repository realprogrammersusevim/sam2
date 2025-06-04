import argparse
import sys
import os
import re # For parsing frame numbers
import numpy as np

# Imports for NPZ to STL conversion
from skimage import measure
from stl import mesh


def convert_npz_frames_to_stl(npz_path, output_stl_path, target_object_id, spacing=(1.0, 1.0, 1.0)):
    """
    Converts segmentation masks for a specific object ID from a series of frames
    in an NPZ file into an STL mesh file.

    The NPZ file is expected to contain arrays like:
    - 'frame_0_masks': (num_objects, H, W) boolean array for frame 0
    - 'frame_0_object_ids': (num_objects,) int array of object IDs for frame 0
    - ... and so on for other frames.
    - 'all_object_ids': (optional) array of all unique object IDs.

    Args:
        npz_path (str): Path to the input NPZ file.
        output_stl_path (str): Path to save the output STL file.
        target_object_id (int): The object ID to extract and mesh.
        spacing (tuple of float): Voxel spacing for the mesh generation
                                  (frame_spacing, height_spacing, width_spacing).
    """
    if not os.path.exists(npz_path):
        print(f"Error: NPZ file not found at {npz_path}")
        sys.exit(1)

    try:
        data_npz = np.load(npz_path)
    except Exception as e:
        print(f"Error loading NPZ file {npz_path}: {e}")
        sys.exit(1)

    frame_keys = sorted([k for k in data_npz.files if k.startswith('frame_') and k.endswith('_masks')])
    
    if not frame_keys:
        print("Error: No 'frame_X_masks' arrays found in the NPZ file.")
        if 'all_object_ids' in data_npz:
             print(f"Available keys in NPZ: {data_npz.files}")
        sys.exit(1)

    # Extract frame numbers and sort them
    frame_numbers = []
    for key in frame_keys:
        match = re.search(r'frame_(\d+)_masks', key)
        if match:
            frame_numbers.append(int(match.group(1)))
    
    if not frame_numbers:
        print("Error: Could not parse frame numbers from NPZ keys.")
        sys.exit(1)
        
    sorted_frame_numbers = sorted(list(set(frame_numbers)))
    print(f"Found {len(sorted_frame_numbers)} frames, from {min(sorted_frame_numbers)} to {max(sorted_frame_numbers)}.")

    object_masks_for_stacking = []
    h, w = -1, -1  # Dimensions, to be determined from the first valid mask

    for frame_num in sorted_frame_numbers:
        mask_key = f'frame_{frame_num}_masks'
        ids_key = f'frame_{frame_num}_object_ids'

        if mask_key not in data_npz or ids_key not in data_npz:
            print(f"Warning: Missing mask or ID data for frame {frame_num}. Skipping.")
            # If H,W known, append empty mask, otherwise this frame can't be processed
            if h != -1 and w != -1:
                 object_masks_for_stacking.append(np.zeros((h, w), dtype=bool))
            continue

        frame_masks_data = data_npz[mask_key]  # Should be (num_obj_in_frame, H, W)
        frame_ids_data = data_npz[ids_key]    # Should be (num_obj_in_frame,)

        if h == -1 and w == -1 and frame_masks_data.ndim == 3 and frame_masks_data.shape[1] > 0 and frame_masks_data.shape[2] > 0:
            h, w = frame_masks_data.shape[1], frame_masks_data.shape[2]
            print(f"Determined mask dimensions (H, W): ({h}, {w})")
        
        if frame_masks_data.ndim != 3 or frame_masks_data.shape[1]!=h or frame_masks_data.shape[2]!=w :
            print(f"Warning: Mask data for frame {frame_num} has unexpected shape {frame_masks_data.shape}. Expected (N, {h}, {w}). Skipping.")
            if h != -1 and w != -1: # Append empty if dimensions known
                 object_masks_for_stacking.append(np.zeros((h, w), dtype=bool))
            continue


        # Find the index of the target object ID in this frame's ID list
        obj_indices = np.where(frame_ids_data == target_object_id)[0]

        if obj_indices.size > 0:
            obj_idx_in_frame = obj_indices[0]
            # Extract the 2D mask for the target object
            # frame_masks_data shape is (num_objects_in_frame, H, W)
            object_mask_2d = frame_masks_data[obj_idx_in_frame, :, :]
            object_masks_for_stacking.append(object_mask_2d.astype(bool))
        else:
            # Target object not found in this frame, append an empty mask
            if h != -1 and w != -1:
                object_masks_for_stacking.append(np.zeros((h, w), dtype=bool))
            else:
                # This case should ideally be avoided if H,W are not yet known
                # and this is the first frame processed.
                print(f"Warning: Target object ID {target_object_id} not found in frame {frame_num}, and H,W not yet set. Cannot create empty mask.")


    if not object_masks_for_stacking:
        print(f"Error: No masks collected for object ID {target_object_id}. Cannot create STL.")
        if 'all_object_ids' in data_npz:
            print(f"Available object IDs in NPZ (from 'all_object_ids'): {data_npz['all_object_ids']}")
        sys.exit(1)
    
    if h == -1 or w == -1:
        print("Error: Could not determine mask dimensions (H, W) from any frame.")
        sys.exit(1)

    # Stack the 2D masks to form a 3D volume
    # The volume will have shape (num_frames, H, W)
    try:
        volume_3d = np.stack(object_masks_for_stacking, axis=0)
    except ValueError as e:
        print(f"Error stacking masks: {e}")
        print("This can happen if masks have inconsistent H, W dimensions across frames.")
        sys.exit(1)

    print(f"Created 3D volume of shape: {volume_3d.shape} for object ID {target_object_id}")

    if not volume_3d.any():
        print(f"Warning: The 3D volume for object ID {target_object_id} is all zeros. No STL will be generated.")
        return

    # Apply Marching Cubes algorithm
    # For boolean data, level=0.5 extracts the surface between False (0) and True (1)
    try:
        verts, faces, normals, values = measure.marching_cubes(
            volume=volume_3d.astype(float),  # Marching cubes expects float data
            level=0.5,
            spacing=spacing  # (frame_axis_spacing, height_axis_spacing, width_axis_spacing)
        )
    except RuntimeError as e:
        print(f"RuntimeError during Marching Cubes: {e}")
        sys.exit(1)
    except ValueError as e: # Often means no surface found
        print(f"ValueError during Marching Cubes: {e}. This might mean no surface was found for object {target_object_id} at level 0.5.")
        sys.exit(1)


    if verts.size == 0 or faces.size == 0:
        print(f"Warning: Marching Cubes did not generate any vertices or faces for object ID {target_object_id}.")
        return

    # Create the STL mesh object
    surface_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f_indices in enumerate(faces):
        for j in range(3):
            surface_mesh.vectors[i][j] = verts[f_indices[j], :]

    # Save the mesh to an STL file
    try:
        surface_mesh.save(output_stl_path)
        print(f"STL file saved successfully to: {output_stl_path}")
    except Exception as e:
        print(f"Error saving STL file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NPZ segmentation frames to an STL mesh.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_npz_file",
        type=str,
        help="Path to the input NPZ file containing frame masks."
    )
    parser.add_argument(
        "output_stl_file",
        type=str,
        help="Path to save the output STL file."
    )
    parser.add_argument(
        "--object_id",
        type=int,
        required=True,
        help="The integer ID of the object to extract and mesh from the NPZ file."
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=('Z', 'Y', 'X'),
        help="Voxel spacing for the mesh (frame_spacing, height_spacing, width_spacing). Default: 1.0 1.0 1.0"
    )

    args = parser.parse_args()

    try:
        print(f"Converting NPZ {args.input_npz_file} to STL {args.output_stl_file} for object ID {args.object_id}")
        convert_npz_frames_to_stl(
            npz_path=args.input_npz_file,
            output_stl_path=args.output_stl_file,
            target_object_id=args.object_id,
            spacing=tuple(args.spacing)
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"Import Error: {e}. Please ensure all dependencies (like numpy, scikit-image, numpy-stl) are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
