import numpy as np
import argparse
import time
import cv2
import People3
import math
import urllib
cnt_up=0
cnt_down=0
pos = 0
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
 
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = cv2.VideoCapture('rtsp://user:user1234@192.168.254.9:554/Streaming/channels/1' )

vs = cv2.VideoCapture('http://user:user1234@192.168.254.7:80/Streaming/channels/1/picture' )
#vs= cv2.VideoCapture(0)
w = vs.get(3)
h = vs.get(4)
frameArea = h*w
areaTH = frameArea/250
line_up = int(2*(w/5))
line_down   = int(3*(w/5))
up_limit =   int(1*(w/5))
down_limit = int(4*(w/5))
line_down_color = (255,0,0)
line_up_color = (0,0,255)
pt1 =  [line_down,0];
pt2 =  [line_down,h];
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [line_up,0];
pt4 =  [line_up,h];
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 =  [up_limit,0];
pt6 =  [up_limit,h];
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [down_limit,0];
pt8 =  [down_limit,h];
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))
persons = []
personSize=6000
max_p_age = 5
pid = 1
font=cv2.FONT_HERSHEY_SIMPLEX
# loop over the frames from the video stream
#stream =urllib.urlopen("http://http://192.168.254.9:8000/*.mpj")
#bytes=''
while True:
	vs = cv2.VideoCapture('http://user:user1234@192.168.254.6:80/Streaming/channels/1/picture' )
       	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	_,frame = vs.read()
	#frame = imutils.resize(frame, width=400)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)),
		0.009843, (224, 224), 127.5)
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]
 
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
 
			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
			if idx==15:
				cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				cy=startY+(int(abs((startY - endY)/2)))
				cx=startX+(int(abs((endX - startX)/2)))
				#cv2.circle(frame,(startX,startY),5,(0,0,255),-1)
				#cv2.circle(frame,(endX,endY),5,(0,255,0),-1)
				cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
				new = True
				inActiveZone= cx in range(up_limit,down_limit)
				for index, p in enumerate(persons):
					dist = math.sqrt((cx - p.getX())**2 + (cy - p.getY())**2)
					if dist <= w/2 and dist <= h/2:
						if inActiveZone:
							new = False
							if p.getX() < line_up and  cx >= up_limit:
								print("[INFO] person going left " + str(p.getId()))
								pos=1
							if p.getX() > line_down and  cx <= down_limit:
								print("[INFO] person going right " + str(p.getId()))
								pos=2
							if p.getlast() > 0:
								if p.getlast()> pos:
									cnt_down+=1
									print("[INFO] person going out " + str(p.getId()))
								elif p.getlast()< pos:
									cnt_up+=1
									print("[INFO] person going in " + str(p.getId()))
							p.updateCoords(cx,cy,pos)
							#print("[INFO] person deatil updated" + str(p.getId()))
							break
						else:
							print("[INFO] person removed " + str(p.getId()))
							persons.pop(index)
				if new == True and inActiveZone:
					print("[INFO] new person " + str(pid))
					p = People3.Person(pid, cx, cy)
					persons.append(p)
					pid += 1
# show the output frame
	str_up = 'IN: '+ str(cnt_up)
	str_down = 'OUT: '+ str(cnt_down)
	frame = cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
	frame = cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
	frame = cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
	frame = cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
	cv2.putText(frame, str_up ,(10,40),font,0.5,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(frame, str_up ,(10,40),font,0.5,(0,0,255),1,cv2.LINE_AA)
	cv2.putText(frame, str_down ,(10,90),font,0.5,(255,255,255),2,cv2.LINE_AA)
	cv2.putText(frame, str_down ,(10,90),font,0.5,(255,0,0),1,cv2.LINE_AA)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
