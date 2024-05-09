import cv2
import os
import copy
import numpy as np

class Preprocessing():
    def __init__(self,df, path):
        self.path=path
        self.df = df 
    def preprocess(self,ID):
        cuis= list(self.df["CUIs"][self.df["ID"] == ID])[0]
        file_path = os.path.join(self.path, ID + ".jpg")
        image = cv2.imread(file_path)
        if('C0040405' in cuis):
            return self.processer_ctscan(image)
        elif('C0024485' in cuis):
            return self.processer_mri(image)
        elif('C0032743' in cuis or 'C1699633' in cuis):
            return self.processer_pet(image)
        elif('C0041618' in cuis):
            return self.processer_ultrasound(image)
        elif('C1306645' in cuis):
            return self.processer_xray(image)
        elif('C0002978' in cuis):
            return self.processer_anogiogram(image)
        else:
            return self.processer_other(image)
        pass
    def processer_ctscan(self,image):
        image_precesser=copy.deepcopy( image)
        image_precesser=self.sharpingAddnoise(image_precesser,1)
        image_precesser=self.maximizeContrast(image_precesser)
#         image_precesser=self.histogram_qualization(image_precesser)
        image_precesser= self.smoothing(image_precesser,0)
        image_precesser=self.thresholding(image_precesser,0,100,200)
        image_precesser= self.canny_edge_detection(image_precesser,50,150)
        image_precesser=self.converGray(image_precesser)
        return image_precesser
    def processer_mri(self,image):
        image_precesser=copy.deepcopy( image)
        image_precesser=self.sharpingAddnoise(image_precesser,1)
        image_precesser=self.maximizeContrast(image_precesser)
#         image_precesser=self.histogram_qualization(image_precesser)
        
        image_precesser= self.smoothing(image_precesser,0)
        image_precesser=self.thresholding(image_precesser,0,100,200)
        image_precesser= self.canny_edge_detection(image_precesser,50,150)
        image_precesser=self.converGray(image_precesser)
        return image_precesser
    def processer_pet(self,image):
        image_precesser=copy.deepcopy( image)
#         image_precesser=self.sharpingAddnoise(image_precesser,1)
        image_precesser=self.sharping(image_precesser)
#         image_precesser=self.maximizeContrast(image_precesser)
#         image_precesser=self.histogram_qualization(image_precesser)
        
#         image_precesser= self.smoothing(image_precesser,0)
        image_precesser=self.thresholding(image_precesser,0,100,200)
#         image_precesser= self.canny_edge_detection(image_precesser,50,150)
#         image_precesser=self.converGray(image_precesser)
        return image_precesser
    def processer_ultrasound(self,image):
        image_precesser=copy.deepcopy( image)
        image_precesser=self.sharpingAddnoise(image_precesser,1)
        image_precesser=self.maximizeContrast(image_precesser)
#         image_precesser=self.histogram_qualization(image_precesser)

#         image_precesser= self.smoothing(image_precesser,0)
#         image_precesser=self.thresholding(image_precesser,0,100,200)
#         image_precesser=self.canny_edge_detection(image_precesser,50,150)
        image_precesser=self.converGray(image_precesser)
        return image_precesser
    def processer_xray(self,image):
        image_precesser=copy.deepcopy( image)
        image_precesser=self.sharpingAddnoise(image_precesser,1)
        image_precesser=self.maximizeContrast(image_precesser)
        image_precesser=self.histogram_qualization(image_precesser)
        
#         image_precesser= self.smoothing(image_precesser,0)
#         image_precesser=self.thresholding(image_precesser,0,100,200)
#         image_precesser= self.canny_edge_detection(image_precesser,50,150)
        image_precesser=self.converGray(image_precesser)
        return image_precesser
    def processer_anogiogram(self,image):
        image_precesser=copy.deepcopy( image)
        image_precesser=self.sharpingAddnoise(image_precesser,1)
#         image_precesser=self.maximizeContrast(image_precesser)
        image_precesser=self.histogram_qualization(image_precesser)
        image_precesser= self.smoothing(image_precesser,0)
    #     image_precesser=self.thresholding(image_precesser,0,100,200)
    #     image_precesser= self.canny_edge_detection(image_precesser,50,150)
        image_precesser=self.converGray(image_precesser)
        return image_precesser
    def processer_other(self,image):
        image_precesser=copy.deepcopy( image)
        image_precesser=self.sharping(image_precesser)
        image_precesser= self.smoothing(image_precesser,0)
        return image_precesser
    def converGray(self,image_gray):
        image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        return image_color
    def smoothing(self,image,options=0,sigma=0):
        match options:
            case 0:
                return cv2.GaussianBlur(image,(5,5),sigma)
            case 1:
                return cv2.blur(image,(5,5))
            case 2:
                return cv2.medianBlur(image,5)
            case 3:
                return cv2.bilateralFilter(image,9,75,75)
            case 4:
                if len(image.shape) <= 2:
                    image=self.converGray(image)
                return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            case default:
                return None
    def histogram_qualization(self,image):
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        equalized_image = cv2.equalizeHist(gray_image) 
        return equalized_image
    def sharping(self,image):
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
        return image_sharp

    # Add Salt and Pepper noise to the original image
    def add_salt_and_pepper_noise(self,image, salt_prob, pepper_prob):
        noisy_image = np.copy(image)
        salt = np.random.rand(*image.shape) < salt_prob
        pepper = np.random.rand(*image.shape) < pepper_prob
        noisy_image[salt] = 255
        noisy_image[pepper] = 0
        return noisy_image.astype(np.uint8)

    # Add Gaussian noise to the original image
    def add_gaussian_noise(self,image, mean, variance):
        row, col = image.shape
        sigma = variance ** 0.5
        gaussian = np.random.normal(mean, sigma, (row, col))
        noisy_image = image + gaussian
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)

    # Perform unsharp masking for image sharpening
    def unsharp_masking(self,image, alpha, beta):
        blurred  =self.smoothing(image,2)
        sharpened = cv2.addWeighted(image, alpha, blurred , -beta, 0)
        return sharpened

    def sharpingAddnoise(self,image,option=0, alpha=4,beta=3):
        if(option==0):
            noisy_image = self.add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01)
            denoised_image = self.smoothing(noisy_image,0)
        else:
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image = image
            noisy_image = self.add_gaussian_noise(image, mean=0, variance= 0.05)
            denoised_image = self.smoothing(noisy_image,0,0.62)
        # Perform image sharpening
        sharpened_image = self.unsharp_masking(denoised_image, alpha=alpha, beta=beta)
        return sharpened_image
        # Display results
    # Otsu's thresholding  cực kì tốt
    def thresholding(self,image,option=0,low_threshold=0, high_threshold=255):
        if option==0:
        # global thresholding 127,255
            _,th = cv2.threshold(image,low_threshold,high_threshold,cv2.THRESH_BINARY)
        if option==1:
        # Otsu's thresholding thương để ngưỡng 0 ,255
            if len(image.shape) > 2:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image
            _,th = cv2.threshold(gray_image,high_threshold,high_threshold,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return th
    def maximizeContrast(self,image, iter = 10):
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        topHat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel, iterations = iter)
        blackHat =cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel, iterations = iter)
        imgGrayscalePlusTopHat = cv2.add(gray_image, topHat)
        imgGraysclaePusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, blackHat)
        return imgGraysclaePusTopHatMinusBlackHat
    

