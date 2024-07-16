# Preprocessing Pipeline


## **def main()**
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
### 3D Series to 2D Slices
---
    * Slice
    * Slice with Specific Order
### Skull Extraction
---
    * Extract Skull Region
### Check Data Behavior
---
    * Check Stastistic
    * Visualize Brain and Skull Extraction Result


## **def mat2nii()**
Change File Format (SciPy)

    1. Load Data
        - MR, CT
        - .mat File

    2. Save Data (NiBabel)
        - MR, CT
        - .nii File


## **def transform()**
Interpolate to (192, 192, 192) + Rotate

    1. Load Data (NiBabel)
        - MR, CT 
        - .nii File

    2. Numpy Array to Troch Tensor

    3. Interpolate (Torch)
        - Trilinear Interpolation
        - (192, 192, 192)

    4. Troch Tensor to Numpy Array

    5. Rotate (NumPy)
        - No.1 ~ No.10 and No.24 ~ No.26
        - Counterclockwise 270 Degree
    
    6. Save Data (NiBabel)
        - MR, CT
        - .nii File 


## **def background()**
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
                - No.1 ~ No.4 and No.1 ~ No.26
                - 12.5%
    
    3. Thresholding

    4. Get Head Mask (ndimage)
        - Get Connective Component
        - Compute Size of Each Component
        - Find Largest Component
        - Slect Largest Component
        - Fill Holes in Mask
    
    5. Apply Mask (NumPy)
        - MR Air Value: 0
        - CT Air Value: -1000

    6. Save Data (NiBabel)
        - MR, CT, Head Mask (HM)
        - .nii File


## **def intensity()**
Clip Intensity
### MR
---
Only Deal with MR Series with Artiface: MR14

    1. Load Data (NiBabel)
        - MR
        - .nii File
    
    2. Sum Up the Maximum Value 
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


## **def strip()**
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


## **def fillhole()**
Fill Holes in Brain Mask

    1. Load Data
        - BR, HM
        - .nii File
    
    2. Apply Mask


## **def fillhole()**
## **def remove()**
## **def n4bias()**
## **def slice()**
## **def specific()**
## **def extract()**
## **def statistic()**
## **def visualize()**