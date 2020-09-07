from gtts import gTTS
import os
import cv2
import numpy as np
from utils.image_transformation import transform
from utils.image_transformation import write_text
from utils.load_data import process_image
import pickle
from sklearn.metrics import adjusted_rand_score
from utils.predict_labels import predict_label_correlation,predict_label_svm
def Image(path):
	frame = cv2.imread(path)

	o_frame = transform(frame)
	label = predict_label_svm(frame)

	annotated_img = write_text(frame,label)
	print(label)
	cv2.imshow('input',annotated_img)
	cv2.imshow('output',o_frame)
	cv2.waitKey(0)

def Video():
	# code for sentence
	word = 'hello '

	cap = cv2.VideoCapture(0)
	while not cap.isOpened():
		cap = cv2.VideoCapture(0)
		cv2.waitKey(1000)
	label = ''
	cntr = 1
	while True:
		flag, frame = cap.read()
		# label = predict_label_svm(frame[100:300,400:600])
		if flag:
			frame = cv2.flip(frame,1)
			cv2.rectangle(frame,(400,100),(600,300),(255,0,0),2)
			annotated_frame = write_text(frame,label)
			cv2.imshow('input',annotated_frame)

		else:
			cv2.waitKey(1000)
		k = cv2.waitKey(5)
		#    tab breaks the flow -> text+audio
		if k == 9:
			print(word)
			audio_feature(word)
			break
		    
		elif k == 27:
			label = predict_label_svm(frame[100:300,400:600])
			# print(label)
			word=word+label

		elif k == ord('s'):
			cv2.imwrite('images/'+str(cntr)+'.jpg',frame[100:300,400:600])
			cntr += 1
			# adding a space with enter 
		elif k == 13:
			label = " "
			word=word+label	

def audio_feature(word):
	language = 'en'
	output = gTTS(text=word, lang=language, slow=False)
	output.save("output.mp3")
	os.system("start output.mp3")

# def test_transformation():
# 	Image('/home/apurv/Personal_Project/GHCI/hand_sign_recognition/images/TRAIN/A/001.jpg')

def test_images_correlation():
	with open('data/train.pickle', 'rb') as handle:
		train = pickle.load(handle)
	with open('data/test.pickle', 'rb') as handle:
		test = pickle.load(handle)
	for test_image in test:
		correlation={}
		best_value = 0
		found=''
		for key in train.keys():

			acc_list = []
			for img in train[key]:
				acc_list.append(adjusted_rand_score(img,test[test_image]))

			correlation[key]=max(acc_list)
			if correlation[key]>best_value:
				best_value=correlation[key]
				found = key
		print('Input: '+test_image+', Detected: '+found+' -> ' + str(test_image==found) + ' -> Confidence: ' + str(best_value))


if __name__ == "__main__":
	# test_transformation()
	#test_images_correlation()
	Video()
	cv2.destroyAllWindows()




    
    
    

