import nibabel as nib
import numpy as np
import imageio  # For video conversion


class NiftiImage:
    """
    A class to handle NIfTI file loading, iteration over slices,
    and conversion to video.
    """

    def __init__(self, file_path):
        """
        Initializes the NiftiImage by loading a NIfTI file.

        Args:
            file_path (str): Path to the .nii or .nii.gz file.
        """
        self.file_path = file_path
        self._img = None
        self._data = None
        self._affine = None
        self._header = None

        self._load_nifti()

        self._current_slice_index = 0

    def _load_nifti(self):
        """Loads the NIfTI file and extracts data, affine, and header."""
        try:
            self._img = nib.load(self.file_path)
            self._data = self._img.get_fdata()
            self._affine = self._img.affine
            self._header = self._img.header
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            raise
        except Exception as e:
            print(f"Error loading NIfTI file '{self.file_path}': {e}")
            raise

    @property
    def data(self):
        """Returns the NIfTI image data as a NumPy array."""
        return self._data

    @property
    def affine(self):
        """Returns the affine transformation matrix."""
        return self._affine

    @property
    def header(self):
        """Returns the NIfTI image header."""
        return self._header

    @property
    def shape(self):
        """Returns the shape of the NIfTI image data."""
        return self._data.shape if self._data is not None else tuple()

    def __iter__(self):
        """Initializes the iterator."""
        self._current_slice_index = 0
        return self

    def __next__(self):
        """
        Returns the next slice of the image.
        For 3D data, iterates along the 3rd axis (axis 2).
        For 2D data, returns the data itself once.
        Raises StopIteration if no more slices.
        """
        if self._data is None:
            raise StopIteration("NIfTI data not loaded.")

        data_dim = len(self.shape)

        if (
            data_dim >= 3
        ):  # Handles 3D or higher-dimensional data, iterating along axis 2
            if self._current_slice_index < self.shape[2]:
                slice_data = self._data[:, :, self._current_slice_index]
                # For >3D data, slice_data will retain subsequent dimensions.
                # e.g., if shape is (X,Y,Z,T), slice_data is (X,Y,T)
                self._current_slice_index += 1
                return slice_data
            else:
                raise StopIteration
        elif data_dim == 2:  # Handles 2D data
            if self._current_slice_index == 0:
                self._current_slice_index += 1
                return self._data
            else:
                raise StopIteration
        else:  # Handles 1D or 0D data
            raise StopIteration(
                "Iteration not supported for 1D or 0D data in this context."
            )

    def __len__(self):
        """
        Returns the number of items that can be iterated over.
        For 3D data, this is the number of slices along the 3rd axis.
        For 2D data, this is 1.
        For other data shapes, it's 0.
        """
        if self._data is None:
            return 0

        data_dim = len(self.shape)
        if data_dim >= 3:
            return self.shape[2]  # Number of slices along the 3rd axis
        elif data_dim == 2:
            return 1  # 2D data iterates once
        else:
            return 0  # Not iterable in a meaningful way for slices

    def get_slice(self, slice_number, slice_axis=2):
        """
        Retrieves a specific slice from the NIfTI data.

        Args:
            slice_number (int): The index of the slice to retrieve.
            slice_axis (int): The axis along which to slice (default is 2).

        Returns:
            numpy.ndarray: The specified slice.

        Raises:
            ValueError: If data is not suitable for slicing (e.g. < 3D for axis > 0).
            IndexError: If slice_number is out of range.
        """
        if self._data is None:
            raise ValueError("NIfTI data not loaded.")

        if not (0 <= slice_axis < len(self.shape)):
            raise ValueError(
                f"Slice axis {slice_axis} is out of range for data shape {self.shape}."
            )

        if not (0 <= slice_number < self.shape[slice_axis]):
            raise IndexError(
                f"Slice number {slice_number} is out of range for axis {slice_axis} (0-{self.shape[slice_axis] - 1})."
            )

        # Create a slicer object
        slicer = [slice(None)] * len(self.shape)  # equivalent to [:, :, :, ...]
        slicer[slice_axis] = slice_number
        return self._data[tuple(slicer)]

    def to_video(self, output_path, fps=10, slice_axis=2):
        """
        Converts 3D NIfTI image slices into a video file.
        This method expects the data to be 3D.

        Args:
            output_path (str): The path to save the output video file (e.g., 'output.mp4').
            fps (int): Frames per second for the video.
            slice_axis (int): The axis along which to take 2D slices for video frames (default is 2).
        """
        if self._data is None:
            print("Error: NIfTI data not loaded. Cannot convert to video.")
            return

        if len(self.shape) != 3:
            print(
                f"Error: Video conversion currently supports 3D data only. Data shape is {self.shape}."
            )
            print(
                "You might need to select a 3D sub-volume from higher-dimensional data."
            )
            return

        if not (0 <= slice_axis < 3):  # For 3D data, axis must be 0, 1, or 2
            print(
                f"Error: slice_axis {slice_axis} is invalid for 3D data. Must be 0, 1, or 2."
            )
            return

        num_slices = self.shape[slice_axis]
        frames = []

        # Normalize the entire 3D volume to 0-255 uint8 for video
        # This ensures consistent brightness/contrast across frames.
        # Using float32 for intermediate calculations to prevent precision loss / overflow
        data_float = self._data.astype(np.float32)
        min_val = np.min(data_float)
        max_val = np.max(data_float)

        normalized_data_uint8 = np.zeros_like(data_float, dtype=np.uint8)
        if max_val > min_val:  # Avoid division by zero if data is flat
            normalized_data_uint8 = (
                255 * (data_float - min_val) / (max_val - min_val)
            ).astype(np.uint8)
        elif (
            max_val == min_val
        ):  # If data is flat, set to a single value (e.g. 0 or 128)
            normalized_data_uint8 = np.full(
                data_float.shape,
                int(min_val) if 0 <= min_val <= 255 else 0,
                dtype=np.uint8,
            )

        for i in range(num_slices):
            if slice_axis == 0:
                slice_data = normalized_data_uint8[i, :, :]
            elif slice_axis == 1:
                slice_data = normalized_data_uint8[:, i, :]
            else:  # slice_axis == 2
                slice_data = normalized_data_uint8[:, :, i]

            # Ensure the slice is 2D (it should be if input is 3D and axis is valid)
            if slice_data.ndim == 2:
                frames.append(slice_data)
            else:
                print(
                    f"Warning: Slice {i} along axis {slice_axis} is not 2D (shape: {slice_data.shape}). Skipping."
                )
                continue

        if not frames:
            print("Error: No valid 2D frames generated for video.")
            return

        try:
            imageio.mimsave(output_path, frames, fps=fps)
            print(f"Video saved to {output_path}")
        except Exception as e:
            print(f"Error saving video: {e}")
            print(
                "Ensure you have a suitable backend for imageio, like imageio-ffmpeg."
            )
            print("You can try: pip install imageio[ffmpeg]")
