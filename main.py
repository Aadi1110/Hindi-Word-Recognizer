from functions import *
import cv2
import numpy as np
from keras.models import load_model
import os

model = load_model('model.h5')

def predict(image):
	chars = segment(image)

	final_size = (28,28)
	kernel=np.ones((2,2),'uint8')
	mapping={
	1:"क", 2:"ख", 4:"ध",
	6:"च", 8:"ज", 10:"ज",
	11:"ट", 12:"ठ", 13:"ड", 14:"ढ",
	16:"त", 17:"थ", 18:"द", 19:"ध", 20:"न",
	21:"प", 22:"फ", 23:"ब", 24:"भ", 25:"म",
	26:"य", 27:"र", 28:"ल", 29:"व", 31:"ष",
	32:"स", 33:"ह", 35:"त्र", 36:"ज्ञ"}

	answer = []

	for c in chars:
		c = cropping(c)
		c = cv2.dilate(c,kernel,iterations=2)
		c = (255-c)/255.0
		cv2.imshow('c',c)
		cv2.waitKey(0)
		c = cv2.resize(c,final_size)
		c = cv2.copyMakeBorder(c, 2, 2, 2, 2, cv2.BORDER_CONSTANT, None, 0)
		c = c.reshape(1,32,32,1)
		pred = model.predict(c)
		answer.append(mapping[np.argmax(pred)])
	return answer

def test():
	image_paths = ['t9.jpeg']
	correct_answers = ['ख']
	score = 0
	multiplication_factor = 5

	for i,image_path in enumerate(image_paths):
		image = cv2.imread(image_path)
		answer = predict(image)
		print(''.join(answer))
		'''
		n = 0
		for j in range(len(answer)):
			if correct_answers[i][j] == answer[j]:
				n+=1
		if(n==len(correct_answers[i])):
			score += len(correct_answers[i])*multiplication_factor

		else:
			score += n*2
		'''
	print('The final score of the participant is',score)

if __name__ == "__main__":
	test()