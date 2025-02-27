# Preprocessing Pipeline

## ***Note***
Specific Data Behavior
### MR
---
    * Artifact
        - 14

    * Especially High Maximum Value
        - 11, 26
### CT
---
    * Artifact
        - 05, 06, 07, 08, 09, 10

    * Especially Bright
        - 02, 05, 06, 07, 08, 09, 10
### Both
---
    * Different Direction
        - 11, 12, 13, 14, 15, 16, 17, 18, 19, 20

## ***def main()***
Contral Overall Pipeline of Preprocessing
### Fundamental Image Process
---
    * Convert File Format
    * Apply Transformation
    * Remove Background
    * Clip Intensity
### Medical Image Process
---
    * MR N4 Bias Correction
    * Extract Brain Region
    * Fill Holes in Brain Mask
    * Remove Useless Area
    * MR Intensity Normalize
    * Extract Skull Region
### Slice
---
    * Slice with Random Order
    * Slice with Specific Order
### Data Behavior
---
    * Check Stastistic
    * Check CT Behavior
    * Visualize Brain and Skull Extraction Result


## ***def convert_format()***
Convert File Format from .mat to .nii

    1. Load Data (SciPy)
        - MR, CT
        - .mat File

    2. Save Data (NiBabel)
        - MR, CT
        - .nii File


## ***def apply_transformation(mode: str | Literal['interpolate', 'padding'])***
Rotate + Shift Intensity + Interpolate or Padding

    1. Load Data (NiBabel)
        - MR, CT 
        - .nii File

    2. Rotate (NumPy)
        - No.1 ~ No.10 and No.21 ~ No.26
        - Counterclockwise 270 Degree

    3. Deal With CT Extreme Case (Python)
        - Shift -1000

    4. Interpolate or Padding 

        Interpolate:

            Compute Z-Axis Scale Factor (Python)
                - Factor = 256 / X or Y-Axis Size

            Interpolate (Torch)
                - Trilinear Interpolation
                - (256, 256, Z-Axis Size * Factor)

        Padding:

            Calculate Number of Padding Pixe (Python)
                - # Pixel = Max(256 - Axis, 0)

            Apply Padding (NumPy)
                - MR: 0
                - CT: -1000

            Crop (NumPy)
                - (256, 256, Z-Axis Size)
    
    5. Save Data (NiBabel)
        - MR, CT
        - .nii File 


## ***def remove_background(otsu: bool = False)***
Remove Background

    1. Load Data (NiBabel)
        - MR, CT
        - .nii File

    2. Remove Rough Background of CT (NumPy)
        - Thresholding with 250 HU
    
    3. Find Background Threshold (NumPy)
        - Flatten CT Data
        - Sort in Ascending Order
        - Get Cumulative Distribution
        - Get Threshold
            - With Otsu's Algorithm
                - Find Best Threshold within 1.25% ~ 12.5%
            - Without Otsu's Algorithm
                - 2.5%
    
    4. Thresholding (NumPy)

    5. Get Head Mask (ndimage)
        - Get Connective Component
        - Compute Size of Each Component
        - Find Largest Component
        - Slect Largest Component

    6. Fill Holes in Mask (ndimage)
        - Closing the Mask Iteratively Along Z-Axis
        - First Element Structure: 17
    
    7. Apply Mask by Setting to Air Value (NumPy)
        - Out of Head Mask
        - MR: 0
        - CT: -1000

    8. Save Data (NiBabel)
        - MR, CT, Head Mask (HM)
        - .nii File


## ***def clip_intensity()***
Clip Intensity
### MR
---
Clip MR14 Intensity

    1. Load Data (NiBabel)
        - MR
        - .nii File
    
    2. Sum Up the Maximum Value (Python)
        - Except for MR14

    3. Clip MR14 Intensity (NumPy)
        - Min: 0
        - Max: Mean Value of Summation
    
    4. Save Data (NiBabel)
        - MR14
        - .nii File
### CT
---
Clip CT Intensity + Deal With Extreme Case

    1. Load Data (NiBabel)
        - CT
        - .nii File

    2. Clip Intensity (NumPy)
        - Min: -1000
        - Max: 3000

    3. Save Data (NiBabel)
        - CT
        - .nii File


## ***def correct_bias()***
N4 Bias Correction

    1. Load Data (SimpleITK)
        - MR
        - .nii File
    
    2. N4 Bias Correction (SimpleITK)

    3. Save Data (SimpleITK)
        - MR
        - .nii File


## ***def extract_brain()***
Extract Brain Region

    1. Load Data (ANTsPy)
        - MR
        - .nii File

    2. Brain Extraction (ANTsPyNet)
        - Modality: T1

    3. Save Data (ANTsPy)
        - Brain Mask (BR)
        - .nii File


## ***def fill_hole()***
Fill Holes in Brain Mask

    1. Load Data (NiBabel)
        - BR, HM
        - .nii File
    
    2. Apply Mask by Setting to Air Value (NumPy)
        - Outside the Head Mask
        - Noise

    3. Thresholding (NumPy)
        - Convert Probabilistic Map to Binary Map

    4. Fill Holes Along Z-Axis (NumPy + ndimage)
        - Process Slice with Whole Zero
            - Create Zero Mask
        - Thresholding
        - Get Connective Component
        - Compute Size of Each Component
        - Find Largest Component
        - Slect Largest Component
        - Closing the Mask
            - Element Structure: 15
        
    5. Stack the Brain Mask (NumPy)
        - (Z, X, Y)

    6. Transpose (NumPy)
        - (X, Y, Z)

    7. Select the Largest Connect Component of Mask

    8. Save Data (NiBabel)
        - BR
        - .nii File


## ***def remove_uselessness()***
Remove Useless Area

    1. Load Data (NiBabel)
        - MR, CT, HM, BR
        - .nii File

    2. Create Buffer List for Cutting Point (NumPy)
        - The Lowest Point in Coronal View

    3. Remove Zero Value in Buffer List + Access Parameter (Python)
        - 0 -> 9999
        - Find Min
        - Find ArgMin
    
    4. Find Appropriate Cutting Point (Python)
        - j < ArgMin
            - Cutting Point = Min
        - Cutting Point ==  9999 
            - Cutting Point = Latest Meaningful Value

    5. Remove Useless Area (NumPy)
        - MR: 0
        - CT: -1000
        - HM: 0

    6. Save Data (NiBabel)
        - MR, CT, HM
        - .nii File


## ***def apply_normalization()***
MR Normalize

    1. Load Data (NiBabel)
        - MR
        - .nii File
    
    2. MR Normalize (NumPy)
        - Z-Score
            - Mean = 0
            - STD = 1
        - Min-Max
            - [0, 1]
        - Linearly Scale
            - [-1, 1]

    3. Save Data (NiBabel)
        - MR
        - .nii File


## ***def extract_skull()***
Extract Skull Region

    1. Load Data (NiBabel)
        - CT, BR
        - .nii File
    
    2. Remove Brain Region (NumPy + ndimage)
        - Erosion Brain Mask Preprocess
            - Element Structure: 13
        - Inside the Brain Mask

    3. Thresholding (Python)
        - 250 HU

    4. Get Skull Mask (ndimage)
        - Get Connective Component
        - Compute Size of Each Component
        - Find Largest Component
        - Slect Largest Component

    5. Fill Holes in Mask
        - Closing the Mask Iteratively Along Z-Axis
        - First Element Structure: 7

    6. Apply Mask by Setting to Air Value (NumPy)
        - Out of Skull Mask

    7. Save Data (NiBabel)
        - Skull (SK)
        - .nii File


## ***def slice_random()threshold: float = 0.075***
Slice with Random Seed

    1. Shuffle File Name List (Python)
        - MR, CT, HM, SK
        - Combine
        - Shuffle
        - Separate

    2. Decide Dataset (Python)
        - 1 ~ 20:   Training
        - 21 ~ 24:  Validation
        - 25 ~ 26:  Test

    3. Load Data (NiBabel)
        - MR, CT, HM, SK
        - .nii File
        - 3D

    4. Find Overall Lower and Upper Bound Index (NumPy)
        - (# Meaningful Pixel) / (# Whole Pixel) > 0

    5. Find Lower and Upper Bound Index (NumPy)
        - (# Meaningful Pixel) / (# Whole Pixel) > 0.075

    6. Slice (NumPy)
        - MR: (256, 256, 7)
        - CT: (256, 256, 1)
        - HM: (256, 256, 1)
        - SK: (256, 256, 1)
        - (X, Y, Z)
    
    7. Transpose (NumPy)
        - (Z, X, Y) 

    8. Rotate (NumPy)
        - Counterclockwise 90 Degree

    9. Remove Blank Area (NumPy)

    10. Save Data (NiBabel)
        - MR, CT, HM, SK
        - .nii File 
        - 2D
    
    11. Print and Save the Order of Slicing (Python)

    12. Reconstruct File Name List (Python)
        - MR, CT, HM, SK
        - Ascending Sort


## ***def slice_ordered(threshold: float = 0.075)***
Slice with Specific Order

    1. Clear File Name List (Python)
        - MR, CT, HM, SK

    2. Load "Slice.txt" (Python)

    3. Get Specific Order (Python)
        - MR, CT, HM, SK
        - Split the Line with Blank Space
        - Select Numerical Part
        - Append to File Name List
    
    4. Slice as Above Procedure


## ***def compute_statistic()***
Check Statistic

    1. Load Data (NiBabel)
        - MR, CT, HM
        - .nii File

    2. Flatten Data (NumPy)

    3. Remove Air Region (Python)
        - Optional
        - Apply Head Mask
        - Select Non-Air Region
    
    4. Save Mean and STD Value of MR and CT (Python)

    5. Print Stastistic (NumPy)
        - File Name
        - Mean 
        - STD
        - Minimum
        - Maximum
    
    6. Print Additional Stastistic (NumPy)
        - Mean of Mean
        - STD of Mean
        - Mean of STD
        - STD of STD


## ***def check_ct()***
Check CT Behavior

    1. Load Data (NiBabel)
        - CT
        - .nii File

    2. Get Soft Tissue Intensity (Python)
        - [X-Axis * (1/2), Y-Axis * (1/2), Z-Axis * (2/3)]

    3. Remove Air Region (NumPy)
        - Apply Head Mask
        - Flatten Data
        - Select Non-Air Region

    4. Print Stastistic (NumPy)
        - Mean 
        - Soft Tissue Intensity


## ***def visualize_extraction()***
Visulize Brain and Skull Extraction Result

    1. Load Data (NiBabel)
        - MR, CT, BR, SK
        - .nii File

    2. Thresholding (NumPy)
        - Brain > 0         -> 1
        - Skulln > -1000    -> 1

    3. Overlap (Python)
        - MR + BR
            - MR + abs(BR * Mean * 5)
        - CT + SK
            - CT + abs(CT * Mean * 3)

    4. Save Data (NiBabel)
        - Visualize (VS)
        - .nii File