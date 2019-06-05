import cv2
import numpy as np 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
count=0
while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.3,5)
	if faces is():
		pass
	else:
		for(x,y,w,h) in faces: 
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			frame = img[y:y+h,x:x+w]
			count+=1
			face = cv2.resize(frame,(275,275))
			face = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			cv2.imwrite('/home/sourabh/Documents/facial_recognition/photos/'+str(count)+'.jpg',frame)

	cv2.imshow(';img',img)
	k=cv2.waitKey(30) & 0xFF
	if k==27:
		break

cv2.release()
cv2.destroyAllWindows()