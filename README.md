# Preprocessing Pipeline

## ***Note***
Specific Data Behavior
### CT
---
    * Artifect
        - 2, 5, 6, 7, 8, 9, 10
### MR & CT
---
    * Different Direction Cases
        - 11, 12, 13, 14, 15, 16, 17, 18, 19, 20

## ***def main()***
Controls the overall preprocessing pipeline

### Fundamental Image Process
---
    * Convert File Format
    * Apply Transformation
    * Remove Background

### Intensity Manipulation
---
    * MR N4 Bias Correction
    * Clip Intensity

### Medical Image Process
---
    * Extract Brain Region
    * Fill Holes in Brain Mask
    * Remove Useless Area
    * Extract Skull Region

### Normalization
---
    * MR Intensity Normalize + Histogram Normalization

### Slice
---
    * Slice with Random Order
    * Slice with Specific Order

## ***def convert_format()***
Convert file format from `.mat` to `.nii`

    1. Load Data                            (SciPy)
    2. Save Data                            (NiBabel)

## ***def apply_transformation()***
Rotate and apply intensity shift

    1. Load Data                            (NiBabel)
    2. Rotate                               (NumPy)
    3. Shift Intensity                      (NumPy)
    4. Pad to 256x256                       (NumPy)
    5. Crop to 256x256
    6. Save Data                            (NiBabel)

## ***def remove_background(otsu: bool = False)***
Remove background using connected components and Otsu thresholding

    1. Load Data                            (NiBabel)
    2. Remove Rough Background              (NumPy)
    3. Determine Threshold                  (Otsu Algr.) (Optional)
    4. Extract Largest Connected Component  (ndimage)
    5. Fill Holes in Mask
    6. Apply Mask to Remove Background
    7. Save Data (NiBabel)

## ***def correct_bias()***
Apply N4 Bias Correction to MR Images

    1. Load Data (SimpleITK)
    2. Apply N4 Bias Correction             (SimpleITK)
    3. Save Data (SimpleITK)

## ***def clip_intensity()***
Clip MR and CT intensities to predefined thresholds

    1. Load Data                            (NiBabel)
    2. Clip MR Intensity 
        * 0 - 99.5% percentile
    3. Clip CT Intensity 
        * -1000 to 3000 HU
    4. Save Data                            (NiBabel)

## ***def extract_brain()***
Extract the brain region using ANTsPyNet

    1. Load Data                            (ANTsPy)
    2. Extract Brain                        (ANTsPyNet)
    3. Save Data                            (NiBabel)

## ***def fill_hole()***
Fill holes in the extracted brain mask

    1. Load Data                            (NiBabel)
    2. Apply Head Mask to Brain Mask
    3. Extract Largest Connected Component  (ndimage)
    4. Fill Holes in Mask
    5. Save Data                            (NiBabel)

## ***def remove_uselessness()***
Remove unnecessary areas from the MR and CT images

    1. Load Data                            (NiBabel)
    2. Identify Lowest Coronal Point
    3. Remove Useless Regions
    4. Save Data                            (NiBabel)

## ***def extract_skull()***
Extract skull region from CT images

    1. Load Data                            (NiBabel)
    2. Apply Erosion to Brain Mask
    3. Threshold CT for Skull Region 
        * HU > 250
    4. Extract Largest Connected Component  (ndimage)
    5. Fill Holes in Mask
    6. Save Data                            (NiBabel)

## ***def apply_normalization()***
Normalize MR images with histogram equalization

    1. Load MR Data                         (NiBabel)
    2. Apply Z-Score Normalization
        * Foreground Mean
        * Foreground STD
    3. Apply Min-Max Normalization
        * [-1, 1]
    4. Apply Histogram Equalization
    5. Save Data                            (NiBabel)

## ***def slice_random(threshold: float = 0.075)***
Randomly slice images into 2D sections

    1. Shuffle File Order
    2. Assign Data to Train/Val/Test
    3. Extract 2D Slices from 3D Volume
    4. Save 2D Data                         (NiBabel)

## ***def slice_ordered(threshold: float = 0.075)***
Slice images based on a predefined order

    1. Load Order from "Slice.txt"
    2. Slice and Save Data as in `slice_random()`

---