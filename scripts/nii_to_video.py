import argparse
import sys
import os
from nifty import NiftiImage


def main():
    parser = argparse.ArgumentParser(
        description="Convert a NIFTI file to a video.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input NIFTI file (e.g., image.nii.gz).",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the output video file (e.g., video.mp4).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the output video. Default is 10.",
    )
    parser.add_argument(
        "--slice_axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Axis along which to slice the NIFTI volume for video frames.\n"
        "  0: Sagittal (usually iterates along the first dimension)\n"
        "  1: Coronal (usually iterates along the second dimension)\n"
        "  2: Axial (usually iterates along the third dimension)\n"
        "Default is 2 (Axial).",
    )

    args = parser.parse_args()

    try:
        print(f"Loading NIFTI file: {args.input_file}")
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input NIFTI file not found: {args.input_file}")

        nifti_image = NiftiImage(args.input_file)
        print("NIFTI file loaded successfully.")

        if len(nifti_image.shape) != 3:
            print(
                f"Error: Input NIFTI file is not 3-dimensional (shape: {nifti_image.shape}). "
                "Video conversion requires 3D data to create a sequence of 2D slices."
            )
            sys.exit(1)

        print(
            f"Converting to video: {args.output_file} with FPS={args.fps}, Slice Axis={args.slice_axis}"
        )
        nifti_image.to_video(
            output_path=args.output_file, fps=args.fps, slice_axis=args.slice_axis
        )
        print(f"Video successfully saved to {args.output_file}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
