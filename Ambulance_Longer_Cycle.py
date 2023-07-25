import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from scipy import signal
import math
import cv2
from skimage.morphology import skeletonize

audio = "C:/Users/rangko/Downloads/ambulancesoundcode/Ambulance_Horn_Combined.wav"  #path = "<path>/Ambulance_Horn_Combined.wav"
x, sr = librosa.load(audio)
mel_bins = 64 # Number of mel bands
n_fft=1024
hop_length=320
window_type ='hann'
fmin = 0
fmax= None
Mel_spectrogram = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window_type, n_mels = mel_bins, power=2.0)
librosa.display.specshow(Mel_spectrogram, sr=sr, x_axis='time', y_axis='mel',hop_length=hop_length)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.savefig('Mel_Spectogram_image.png')

image = cv2.imread("C:/Users/rangko/Downloads/ambulancesoundcode/Mel_Spectogram_image.png")  #path = "<path>/Mel_Spectogram_image.png"
cv2.imshow("Mel_Spectogram",image)
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Grayscale Image",grayscale)
ret, thresh = cv2.threshold(grayscale, 100, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary Image 1",thresh)
x_s=80
y_e=332
x_e=511
y_s=37
cropped_img_mel = thresh[y_s:y_e, x_s:x_e]
white=np.sum(cropped_img_mel == 255)
if white==0:
	print(False)#,"Mel itself")
	exit(0)

mel_spectrogram_db = librosa.power_to_db(Mel_spectrogram, ref=np.max)
librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='log',hop_length=hop_length)
#plt.colorbar(format='%+2.0f dB')
plt.title('Log Mel spectrogram')
plt.tight_layout()
plt.savefig('Log_Mel_Spectogram_image.png')

image = cv2.imread("C:/Users/rangko/Downloads/ambulancesoundcode/Log_Mel_Spectogram_image.png")   #path = "<path>/Log_Mel_Spectogram_image.png"
cv2.imshow("Log_Mel_Spectogram",image)
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image",grayscale)
ret, thresh1 = cv2.threshold(grayscale, 170, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary Image",thresh1)
y1=75
y2=150
x1=88
x2=517
#plt.figure()
#plt.imshow(skeleton,cmap=plt.cm.gray)
cropped_img = thresh1[y1:y2, x1:x2]
cv2.imshow("Cropped Image",cropped_img)
skeleton = skeletonize(cropped_img)
plt.figure()
plt.imshow(skeleton,cmap=plt.cm.gray)
plt.savefig('Skeleton_image.png')

def next_locations_up(y_down,x_down,part_no):
	y,x=0,0
	x_low=x_down+20
	x_high=x_down+125
	y_low=0
	y_high=36
	white_down=np.sum(img[y_low:y_high,x_low:x_high] == 255)
	if white_down==0:
		return (y,x)
	indices_white_inside=np.where(img[y_low:y_high,x_low:x_high] == [255])
	white_y_inside=indices_white_inside[0]
	white_x_inside=indices_white_inside[1]
	min_y=np.min(white_y_inside)
	x=white_x_inside[0]+x_low
	y=min_y
	count1=np.sum(img[y:y_down,x_down:x] == 255)
	#print("Count:",count1)
	if 3<count1<250:
		print("Up Coordinates:",(y,x),"Count:",count1)
		return (y,x)
	return (0,0)

def next_locations_down(y_up,x_up,part_no,count):
	y,x=0,0
	x_low=x_up+20
	if part_no==3 and count==2:
		x_high=np.shape(img)[1]
	else:
		x_high=x_up+100
	y_low=40
	y_high=np.shape(img)[0]
	white_down=np.sum(img[y_low:y_high,x_low:x_high] == 255)
	if white_down==0:
		return (y,x)
	indices_white_inside=np.where(img[y_low:y_high,x_low:x_high] == [255])
	white_y_inside=indices_white_inside[0]
	white_x_inside=indices_white_inside[1]
	max_y=np.max(white_y_inside)
	for j in range(len(white_y_inside)):
			if white_y_inside[j]==max_y:
				x=white_x_inside[j]+x_low
	#print(white_x_inside)
	y=max_y+y_low
	count1=np.sum(img[y_up:y,x_up:x] == 255)
	#print("Count:",count1)
	if 3<count1<250:
		print("Down Coordinates:",(y,x),"Count:",count1)
		return (y,x)
	return (0,0)

def next_locations(y_down1,x_down1,part_no):
	count=1
	(y,x)=next_locations_up(y_down1,x_down1,part_no)
	if (y,x)==(0,0):
		return False
	count+=1
	(y,x)=next_locations_down(y,x,part_no,count)
	if (y,x)==(0,0):
		return False
	count+=1
	if part_no!=3:
		(y,x)=next_locations_up(y,x,part_no)
		if (y,x)==(0,0):
			return False
		count+=1
	if part_no!=3 and count==4:
		return True
	if part_no==3 and count==3:
		return True
	return False

def checking_white(min_y,x_val,part_no):
	for i in x_val:
		#print("x:",i,"y:",min_y)
		x_min=i+(part_no-1)*x_length+20
		x_max=i+(part_no-1)*x_length+100
		y_min=40
		y_max=np.shape(img)[0]
		#print(x_min,x_max)
		#cv2.imshow("Down Part",img[y_min:y_max,x_min:x_max])
		white_down=np.sum(img[y_min:y_max,x_min:x_max] == 255)
		#print("White Down1",white_down)
		if white_down==0:
			continue
		indices_white_inside=np.where(img[y_min:y_max,x_min:x_max] == [255])
		white_y_inside=indices_white_inside[0]
		white_x_inside=indices_white_inside[1]
		max_y=np.max(white_y_inside)
		x_point=[]
		for j in range(len(white_y_inside)):
			if white_y_inside[j]==max_y:
				x_point.append(white_x_inside[j]+x_min)
		#print(white_x_inside)
		y_point=max_y+y_min
		#print("y_max:",y_point,"x_val",x_point)
		for k in x_point:
			count1=np.sum(img[min_y:y_point,i:k] == 255)
			#print("Count:",count1)
			if 3<count1<250:
				print("1st up coordinates:",(min_y,i+(part_no-1)*x_length))
				print("1st down coordinates:",(y_point,k),"count: ",count1)
				val=next_locations(y_point,k,part_no)
				if val:
					return val
	return False

image = cv2.imread("C:/Users/rangko/Downloads/ambulancesoundcode/Skeleton_image.png")   #path = "<path>/Skeleton_image.png"
cv2.imshow("Skeletonized Image",image)
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Grayscale', grayscale)
ret, thresh1 = cv2.threshold(grayscale, 170, 255, cv2.THRESH_BINARY)
y1=199
y2=285
x1=80
x2=576
#plt.figure()
#plt.imshow(skeleton,cmap=plt.cm.gray)
cropped_img = thresh1[y1:y2, x1:x2]
cv2.imshow("Cropped",cropped_img)
kernel = np.ones((3,3),np.uint8)
#opening = cv2.morphologyEx(cropped_img, cv2.MORPH_OPEN, kernel)
#cv2.imshow("Opening",opening)
contours, hierarchy = cv2.findContours(cropped_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print(contours)
contours1=[]
for contour in contours:
	if len(contour)>5: #and cv2.contourArea(contour)>50:
		contours1.append(contour)
global img
img = np.zeros((86,496), dtype = np.uint8)
cv2.drawContours(img, contours1, -1, (255,255,255), 1)
#print("Contours1:\n",contours1)
cv2.imshow("Contours and removing smaller contours",img)
global x_length
x_length=np.shape(img)[1]//4

y=np.shape(img)[0]
im1=img[0:y,0:x_length]
im2=img[0:y,x_length:(2*(x_length))]
im3=img[0:y,(2*(x_length)):(3*(x_length))]
im4=img[0:y,(3*(x_length)):(np.shape(img)[1])]
#print("im1 shape",np.shape(im1))
#cv2.imshow("Contours1",im1[0:24,])
#cv2.imshow("Contours2",im2)
#cv2.imshow("Contours3",im3)
#cv2.imshow("Contours4",im4)
white_pix = np.sum(img == 255)
white_pix_top = np.sum(img[0:36,] == 255)

if white_pix<10 or white_pix_top<2:
	print(False)
white_pix1=np.sum(im1[0:36,] == 255)
white_pix2=np.sum(im2[0:36,] == 255)
white_pix3=np.sum(im3[0:36,] == 255)
if white_pix1==0 and white_pix2==0 and white_pix3==0:
	print(False)
TorF_val=False
if white_pix1 != 0:
	indices_white=np.where(im1[0:36,] == [255])
	white_y1=indices_white[0]
	white_x1=indices_white[1]
	min_y=np.min(white_y1)
	x_val=[]
	i=0
	while white_y1[i]==min_y:
		x_val.append(white_x1[i])
		i+=1
	TorF_val=checking_white(min_y,x_val,1)
#print("True or False val",TorF_val)
TorF_val1=False
if TorF_val==False and white_pix2 !=0:
	indices_white=np.where(im2[0:36,] == [255])
	white_y1=indices_white[0]
	white_x1=indices_white[1]
	min_y=np.min(white_y1)
	x_val=[]
	i=0
	while white_y1[i]==min_y:
		x_val.append(white_x1[i])
		i+=1
	TorF_val1=checking_white(min_y,x_val,2)
TorF_val2=False
if TorF_val==False and TorF_val1==False and white_pix3 !=0:
	indices_white=np.where(im3[0:36,] == [255])
	white_y1=indices_white[0]
	white_x1=indices_white[1]
	min_y=np.min(white_y1)
	x_val=[]
	i=0
	while white_y1[i]==min_y:
		x_val.append(white_x1[i])
		i+=1
	TorF_val2=checking_white(min_y,x_val,3)
if TorF_val or TorF_val1 or TorF_val2:
	print(True)
else:
	print(False)
cv2.waitKey(0)
cv2.destroyAllWindows()
