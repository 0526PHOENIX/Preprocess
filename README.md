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
    * Direction
        - 11, 12, 13, 14, 15, 16, 17, 18, 19, 20

## ***def main()***
Contral Overall Pipeline of Preprocessing
### Fundamental Process
---
    * Change File Format
    * Interpolate + Rotate
    * Remove Background
    * Clip Intensity
### Brain Extraction
---
    * Extract Brain Region
    * Fill Holes in Brain Mask
    * Remove Useless Area
    * N4 Bias Correction
    * Normalize
### Skull Extraction
---
    * Extract Skull Region
### 3D Series to 2D Slices
---
    * Slice
    * Slice with Specific Order
### Check Data Behavior
---
    * Check Stastistic
    * Check CT Behavior
    * Visualize Brain and Skull Extraction Result


## ***def mat2nii()***
Change File Format

    1. Load Data (SciPy)
        - MR, CT
        - .mat File

    2. Save Data (NiBabel)
        - MR, CT
        - .nii File


## ***def transform()***
Interpolate + Rotate

    1. Load Data (NiBabel)
        - MR, CT 
        - .nii File

    2. Interpolate (Torch)
        - Trilinear Interpolation
        - (192, 192, 192)

    3. Rotate (NumPy)
        - No.1 ~ No.10 and No.21 ~ No.26
        - Counterclockwise 270 Degree
    
    4. Save Data (NiBabel)
        - MR, CT
        - .nii File 


## ***def background()***
Remove Background

    1. Load Data (NiBabel)
        - MR, CT
        - .nii File
    
    2. Find Background Threshold (NumPy)
        - Flatten MR Data
        - Sort in Ascending Order
        - Get Cumulative Distribution
        - Get Threshold
            - With CT Artifact
                - No.5 ~ No.10
                - 20.0%
            - Without CT Artifact
                - No.1 ~ No.4 and No.11 ~ No.26
                - 12.5%
    
    3. Thresholding (NumPy)

    4. Get Head Mask (ndimage)
        - Get Connective Component
        - Compute Size of Each Component
        - Find Largest Component
        - Slect Largest Component
        - Fill Holes in Mask
            - Dilation: (25, 25, 25)
            - Erosion:  (25, 25, 25)
    
    5. Apply Mask by Setting to Air Value (NumPy)
        - Out of Head Mask

    6. Save Data (NiBabel)
        - MR, CT, Head Mask (HM)
        - .nii File


## ***def intensity()***
Clip Intensity
### MR
---
Only Deal with MR Series with Artiface: MR14

    1. Load Data (NiBabel)
        - MR
        - .nii File
    
    2. Sum Up the Maximum Value (Python)
        - Except for MR14

    4. Clip Intensity (NumPy)
        - MR14
        - Min: 0
        - Max: Mean Value of Summation
    
    5. Save Data (NiBabel)
        - MR14
        - .nii File
### CT
---
    1. Load Data (NiBabel)
        - CT
        - .nii File

    2. Clip Intensity (NumPy)
        - Min: -1000
        - Max: 3000

    3. Save Data (NiBabel)
        - CT
        - .nii File


## ***def strip()***
Extract Brain Region

    1. Load Data (ANTsPy)
        - MR
        - .nii File

    2. Brain Extraction (ANTsPyNet)
        - Modality
            -T1

    3. Save Data (ANTsPy)
        - Brain Mask (BR)
        - .nii File


## ***def fillhole()***
Fill Holes in Brain Mask

    1. Load Data (NiBabel)
        - BR, HM
        - .nii File
    
    2. Apply Mask by Setting to Air Value (NumPy)
        - Outside the Head Mask
        - Noise

    3. Fill Holes or Remove Small Component in Every Slice (NumPy + ndimage)
        - Process Slice with Whole Zero
            - Create Zero Mask
        - Thresholding
        - Get Connective Component
        - Compute Size of Each Component
        - Find Largest Component
        - Slect Largest Component
        - Fill Holes in Mask
            - Dilation: (15, 15)
            - Erosion:  (15, 15)
        
    4. Stack the Brain Mask (NumPy)
        - (Z, X, Y)

    5. Transpose (NumPy)
        - (X, Y, Z)

    6. Save Data (NiBabel)
        - BR
        - .nii File


## ***def remove()***
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


## ***def n4bias()***
N4 Bias Correction

    1. Load Data (SimpleITK)
        - MR
        - .nii File
    
    2. N4 Bias Correction (SimpleITK)

    3. Save Data (SimpleITK)
        - MR
        - .nii File


## ***def normalize()***
MR Normalize

    1. Load Data (NiBabel)
        - MR, CT, HM
        - .nii File
    
    2. MR Normalize (NumPy)
        - Z-Score 
        - Min-Max
            - [0, 1]
        - Linearly Scale
            - [-1, 1]

    3. CT Normalize (Numpy)
        - No.02 and No.05 ~ No.10
        - Shift -1000 
            - Non-Air Region

    4. Save Data (NiBabel)
        - MR, CT
        - .nii File


## ***def extract()***
Extract Skull Region

    1. Load Data (NiBabel)
        - CT, BR
        - .nii File
    
    2. Apply Mask by Setting to Air Value (NumPy + ndimage)
        - Brain Mask Preprocess
            - Erosion: (13, 13, 13)
        - Inside the Brain Mask
    
    3. Find Skull Threshold (Python)
        - Remove Air Region
        - Threshold = (Mean * 1 + STD * 0)

    4. Thresholding (Python)

    5. Get Skull Mask (ndimage)
        - Get Connective Component
        - Compute Size of Each Component
        - Find Largest Component
        - Slect Largest Component
        - Fill Holes in Mask
            - Dilation: (5, 5, 5)
            - Erosion:  (5, 5, 5)

    6. Apply Mask by Setting to Air Value (NumPy)
        - Out of Skull Mask

    7. Save Data (NiBabel)
        - Skull (SK)
        - .nii File


## ***def slice()***
Slice

    1. Shuffle File Name List (Python)
        - MR, CT, HM, BR, SK
        - Combine
        - Shuffle
        - Separate

    2. Decide Dataset (Python)
        - 1 ~ 20:   Training
        - 21 ~ 24:  Validation
        - 25 ~ 26:  Test

    3. Load Data (NiBabel)
        - MR, CT, HM, BR, SK
        - .nii File
        - 3D

    4. Find Lower and Upper Bound Index (NumPy)
        - (# Meaningful Pixel) / (# Whole Pixel) > 0.075

    5. Slice (NumPy)
        - MR: (192, 192, 7)
        - CT: (192, 192, 1)
        - HM: (192, 192, 1)
        - (X, Y, Z)
    
    6. Transpose (NumPy)
        - (Z, X, Y) 

    7. Rotate (NumPy)
        - Counterclockwise 90 Degree

    8. Save Data (NiBabel)
        - MR, CT, HM, BR, SK
        - .nii File 
        - 2D
    
    9. Print and Save the Order of Slicing (Python)

    10. Reconstruct File Name List (Python)
        - MR, CT, HM, BR, SK
        - Ascending Sort


## ***def specific()***
Slice with Specific Order

    1. Clear File Name List (Python)
        - MR, CT, HM, BR, SK

    2. Load "Slice.txt" (Python)

    3. Get Specific Order (Python)
        - MR, CT, HM, BR, SK
        - Split the Line with Blank Space
        - Select Numerical Part
        - Append to File Name List
    
    4. Slice as Above Procedure


## ***def statistic()***
Check Statistic

    1. Load Data (NiBabel)
        - MR, CT
        - .nii File

    2. Remove Air Region (Python)
        - Optional
    
    3. Save Mean and STD Value of MR and CT (Python)

    4. Print Stastistic (NumPy + stats)
        - File Name
        - Mean 
        - STD
        - Minimum
        - Maximum
        - Skewness
        - Kurtosis
    
    5. Print Additional Stastistic (NumPy)
        - Mean of Mean
        - STD of Mean
        - Mean of STD
        - STD of STD


## ***def checkct()***
Check CT Behavior

    1. Load Data (NiBabel)
        - CT
        - .nii File

    2. Get Soft Tissue Intensity (Python)
        - [96, 96, 144]

    3. Remove Air Region (Python)

    4. Print Stastistic (NumPy)
        - Mean 
        - Soft Tissue Intensity


## ***def visualize()***
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
