import cv2
import numpy as np
from PIL import Image
import requests

# Difference of Gaussians applied to img input
def dog(img,size=(0,0),k=1.6,sigma=0.5,gamma=1):
	img1 = cv2.GaussianBlur(img,size,sigma)
	img2 = cv2.GaussianBlur(img,size,sigma*k)
	return (img1-gamma*img2)

# Threshold the dog image, with dog(sigma,k) > 0 ? 1(255):0(0)
def edge_dog(img,sigma=0.5,k=200,gamma=0.98):
	aux = dog(img,sigma=sigma,k=k,gamma=0.98)
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] > 0):
				aux[i,j] = 255
			else:
				aux[i,j] = 0
	return aux

# garygrossi xdog version
def xdog_garygrossi(img,sigma=0.5,k=200, gamma=0.98,epsilon=0.1,phi=10):
	aux = dog(img,sigma=sigma,k=k,gamma=gamma)/255
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] >= epsilon):
				aux[i,j] = 1
			else:
				ht = np.tanh(phi*(aux[i][j] - epsilon))
				aux[i][j] = 1 + ht
	return aux*255

def hatchBlend(image):
	xdogImage = xdog(image,sigma=1,k=200, gamma=0.5,epsilon=-0.5,phi=10)
	hatchTexture = cv2.imread('./imgs/hatch.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
	hatchTexture = cv2.resize(hatchTexture,(image.shape[1],image.shape[0]))
	alpha = 0.120
	return (1-alpha)*xdogImage + alpha*hatchTexture

# version of xdog inspired by article
def xdog(img,sigma=0.5,k=1.6, gamma=1,epsilon=1,phi=1):
	aux = dog(img,sigma=sigma,k=k,gamma=gamma)/255
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] < epsilon):
				aux[i,j] = 1*255
			else:
				aux[i,j] = 255*(1 + np.tanh(phi*(aux[i,j])))
	return aux

def to_sketch(img_orig, sigma=0.3, k=4.5, gamma=0.95, epsilon=-1, phi=10e15, area_min=2):
    img_cnts = []
    img = cv2.cvtColor(np.array(img_orig), cv2.COLOR_RGB2GRAY)
    img_xdog = xdog(img, sigma=sigma, k=k, gamma=gamma, epsilon=epsilon, phi=phi).astype(np.uint8)
    new_img = np.zeros_like(img_xdog)
    thresh = cv2.threshold(img_xdog, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area > area_min:
            img_cnts.append(c)

    return Image.fromarray(255 - cv2.drawContours(new_img, img_cnts, -1, (255,255,255), -1))
