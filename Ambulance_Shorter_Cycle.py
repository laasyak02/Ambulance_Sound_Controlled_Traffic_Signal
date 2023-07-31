import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from scipy import signal
import math
import cv2
from skimage.morphology import skeletonize

audio = 'C:/Users/rangko/Downloads/ambulancesoundcode/Ambulance_in_Traffic(Cropped).wav'     #path = "<path>/Ambulance_in_Traffic(Cropped).wav"
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

image1 = cv2.imread("C:/Users/rangko/Downloads/ambulancesoundcode/Mel_Spectogram_image.png")    #path = "<path>/Mel_Spectogram_image.png"
cv2.imshow("Mel_Spectogram",image1) 
grayscale1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Grayscale Image",grayscale)
ret, thresh = cv2.threshold(grayscale1, 100, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary Image 1",thresh)
x_s=80
y_e=332
x_e=511
y_s=37
cropped_img_mel = thresh[y_s:y_e, x_s:x_e]
white=np.sum(cropped_img_mel == 255)
if white==0:
	print(False,"Mel itself")
	exit(0)

mel_spectrogram_db = librosa.power_to_db(Mel_spectrogram, ref=np.max)
librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='log',hop_length=hop_length)
#plt.colorbar(format='%+2.0f dB')
plt.title('Log Mel spectrogram')
plt.tight_layout()
plt.savefig('Log_Mel_Spectogram_image.png')

image = cv2.imread("C:/Users/rangko/Downloads/ambulancesoundcode/Log_Mel_Spectogram_image.png")    #path = "<path>/Log_Mel_Spectogram_image.png"
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

'''
Function Name: next_locations_up
Input parameters: Coordinates of the current minima and the part number.
Purpose: This function finds the following maxima to this minima. 
Output parameters: Returns the coordinates of the maxima point, if found, else returns (0,0)
'''
def next_locations_up(y_down,x_down,part_no):
	y,x=0,0
	x_low=x_down+8
	x_high=x_down+32
	y_low=0
	y_high=54
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
	if 0<count1<50:
		print("Up Coordinates:",(y,x),"Count:",count1)
		return (y,x)
	return (0,0)

'''
Function Name: next_locations_down
Input parameters: Coordinates of the current maxima, the part number, the number of maximas and minimas found until now, and the number of maximas and minimas needed to be found totally.
Purpose: This function finds the following minima to this maxima.
Output parameters: Returns the coordinates of the minima point, if found, else returns (0,0)
'''
def next_locations_down(y_up,x_up,part_no,count,count_limit):
	y,x=0,0
	x_low=x_up+8
	x_high=x_up+33
	if count==count_limit-1 and x_high>np.shape(img)[1]:
		x_high=np.shape(img)[1]
	y_low=60
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
	if 3<count1<100:
		print("Down Coordinates:",(y,x),"Count:",count1)
		return (y,x)
	return (0,0)

'''
Function Name: next_locations
Input parameters: Coordinates of the first minima found and the part number associated with it.
Purpose: This function calculates the number of continuous maximas and minimas found, i.e. it calculates the number of periodic cycles found. This is done with the help of next_locations_up and next_locations_down functions.
Output parameters: Returns True if the required number of periodic cycles are found and False otherwise.
'''
def next_locations(y_down1,x_down1,part_no):
	count=1
	y=y_down1
	x=x_down1
	if part_no==1:
		count_limit=19
	if part_no==2:
		count_limit=17
	if part_no==3:
		count_limit=15
	if part_no==4:
		count_limit=13
	if part_no==5:
		count_limit=11
	while count<count_limit:
		(y,x)=next_locations_up(y,x,part_no)
		if (y,x)==(0,0):
			return False
		count+=1
		(y,x)=next_locations_down(y,x,part_no,count,count_limit)
		if (y,x)==(0,0):
			return False
		count+=1
	
	if count==count_limit:
		return True
	else:
		return False

'''
Function Name: checking_white
Input parameters: Coordinates of the first maxima found and the part number in which it is detected.
Purpose: This function finds the following minima to this maxima. If it is found, the coordinates of this point is sent as the input to the next_locations function along with the part number.
Output parameters: Returns True if the required number of periodic cycles are found and False otherwise.
'''
def checking_white(min_y,x_val,part_no):
	for i in x_val:
		x_min=i+(part_no-1)*x_length+8
		x_max=i+(part_no-1)*x_length+33
		y_min=60
		y_max=np.shape(img)[0]
		white_down=np.sum(img[y_min:y_max,x_min:x_max] == 255)
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
			if 3<count1<100:
				print("1st up coordinates:",(min_y,i+(part_no-1)*x_length))
				print("1st down coordinates:",(y_point,k),"count: ",count1)
				val=next_locations(y_point,k,part_no)
				if val:
					return val
	return False

image = cv2.imread("C:/Users/rangko/Downloads/ambulancesoundcode/Skeleton_image.png")    #path = "<path>/Skeleton_image.png"
cv2.imshow("Skeletonized Image",image)
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(grayscale, 170, 255, cv2.THRESH_BINARY)
y1=199
y2=285
x1=80
x2=576
cropped_img = thresh1[y1:y2, x1:x2]
cv2.imshow("Cropped",cropped_img)
kernel = np.ones((3,3),np.uint8)
contours, hierarchy = cv2.findContours(cropped_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
global img
img = np.zeros((86,496), dtype = np.uint8)
cv2.drawContours(img, contours, -1, (255,255,255), 1)
cv2.imshow("Contours",img)

global x_length
x_length=np.shape(img)[1]//12
y=np.shape(img)[0]
im1=img[0:y,0:x_length]
im2=img[0:y,x_length:(2*(x_length))]
im3=img[0:y,(2*(x_length)):(3*(x_length))]
im4=img[0:y,(3*(x_length)):(4*(x_length))]
im5=img[0:y,(4*(x_length)):(5*(x_length))]
white_pix = np.sum(img == 255)
white_pix_top = np.sum(img[0:54,] == 255)

if white_pix<10 or white_pix_top<2:
	print(False)
white_pix1=np.sum(im1[0:54,] == 255)
white_pix2=np.sum(im2[0:54,] == 255)
white_pix3=np.sum(im3[0:54,] == 255)
white_pix4=np.sum(im4[0:54,] == 255)
white_pix5=np.sum(im5[0:54,] == 255)
if white_pix1==0 and white_pix2==0 and white_pix3==0 and white_pix4==0 and white_pix5==0:
	print(False)
TorF_val=False
if white_pix1 != 0:
	indices_white=np.where(im1[0:54,] == [255])
	white_y1=indices_white[0]
	white_x1=indices_white[1]
	min_y=np.min(white_y1)
	x_val=[]
	i=0
	while white_y1[i]==min_y:
		x_val.append(white_x1[i])
		i+=1
	TorF_val=checking_white(min_y,x_val,1)
TorF_val1=False
if TorF_val==False and white_pix2 !=0:
	indices_white=np.where(im2[0:54,] == [255])
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
	indices_white=np.where(im3[0:54,] == [255])
	white_y1=indices_white[0]
	white_x1=indices_white[1]
	min_y=np.min(white_y1)
	x_val=[]
	i=0
	while white_y1[i]==min_y:
		x_val.append(white_x1[i])
		i+=1
	TorF_val2=checking_white(min_y,x_val,3)
TorF_val3=False
if TorF_val==False and TorF_val1==False and TorF_val2==False and white_pix4 !=0:
	indices_white=np.where(im4[0:54,] == [255])
	white_y1=indices_white[0]
	white_x1=indices_white[1]
	min_y=np.min(white_y1)
	x_val=[]
	i=0
	while white_y1[i]==min_y:
		x_val.append(white_x1[i])
		i+=1
	TorF_val3=checking_white(min_y,x_val,4)
TorF_val4=False
if TorF_val==False and TorF_val1==False and TorF_val2==False and TorF_val3==False and white_pix5 !=0:
	indices_white=np.where(im5[0:54,] == [255])
	white_y1=indices_white[0]
	white_x1=indices_white[1]
	min_y=np.min(white_y1)
	x_val=[]
	i=0
	while white_y1[i]==min_y:
		x_val.append(white_x1[i])
		i+=1
	TorF_val4=checking_white(min_y,x_val,5)

if TorF_val or TorF_val1 or TorF_val2 or TorF_val3 or TorF_val4:
	print(True)
else:
	print(False)
cv2.waitKey(0)
cv2.destroyAllWindows()
