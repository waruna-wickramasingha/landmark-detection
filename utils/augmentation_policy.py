import mimetypes
from pathlib import Path
import os
import random
import shutil
from tempfile import TemporaryDirectory
import albumentations as A
import cv2
import matplotlib.pyplot as plt

random.seed(13)

def doAugmentation(inputDir: str, outputDirRoot: str, augmentationPipeline, numberOfTargetSamples=100):
    """
    This function is to perform image augmentations for the images present at the 'inputDir' 
    into the root location as specified at 'outputDirRoot' using the function 'augmentationPipeline' 
    which takes in an image and output an augmented image. The new directory will be created with the same prefix
    as the input directory and it will contain all the original images plus a number of augmented images 
    such that 'numberOfTargetSamples' is reached

    """
    inputDirPath = Path(inputDir)
    originalImagePaths = []
    im_ext = [k for k,v in mimetypes.types_map.items() if 'image/' in v]

    for x in inputDirPath.iterdir():
        if x.suffix.lower() in im_ext:
            originalImagePaths.append(x)
    
    origLMDir = inputDir.split(os.sep)[-1]
    
    try:
        augLMDir = TemporaryDirectory(dir=outputDirRoot, prefix="{}_Aug_".format(origLMDir))
    except:
        raise Exception("Error creating temp dir for augmentations")
    
    print("Original Directory={}, Augmented Directory={}".format(inputDir, augLMDir.name))
    
    for im in originalImagePaths:
        imageName = str(im).split(os.sep)[-1]
        outputImagePath = augLMDir.name + os.sep + imageName
        
        try:
            shutil.copy(im, outputImagePath)
        except:
            raise Exception("Failed to copy original file {} to {}".format(im, outputImagePath))
    
    augCandidates = random.choices(originalImagePaths, k=(numberOfTargetSamples-len(originalImagePaths)))
    aug_ind = 0
    
    for impath in augCandidates:
        im = cv2.imread(str(impath))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
        augmentedImage = augmentationPipeline(im)
        imageName, imageExt = str(impath).split(os.sep)[-1].split('.')
        outputImagePath = augLMDir.name + os.sep + 'aug' + str(aug_ind) + '_' + imageName + '.' + imageExt
        aug_ind += 1
        try:
            cv2.imwrite(outputImagePath, augmentedImage)
        except:
            raise Exception("Failed to save augmented image to {}".format(outputImagePath))
        
    return augLMDir

transform = A.Compose(
        [A.RandomRotate90(),
        A.RandomBrightnessContrast(brightness_limit=0.8, contrast_limit=0.4, p=0.7),
        A.CLAHE(p=0.7),
        A.HorizontalFlip(p=0.7),
        A.Transpose(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.75),
        A.Blur(blur_limit=3)])

def augmentationPipelineForWaruna(image):
    return transform(image=image)['image']

if __name__ == "__main__":
    inputDir = TemporaryDirectory(dir='/content/sample_data', prefix='waruna')
    doAugmentation(inputDir.name, '/content/sample_data', augmentationPipelineForWaruna, 100)