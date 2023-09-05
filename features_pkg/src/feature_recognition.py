#!/usr/bin/python
from ultralytics import YOLO
import cv2  
import time
from colorthief import ColorThief
import dlib
import numpy as np
from PIL import Image
from roboflow import Roboflow
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as imgmsg
from features_pkg.srv import Features

class FeaturesRecognition:

    def __init__(self):
        self.personModel = YOLO("yolov8m.pt")
        self.keypointsModel = YOLO('yolov8n-pose.pt')
        rf = Roboflow(api_key="NSkF2YE2ufLzrNUiswiA")
        project = rf.workspace().project("mask-detection-m3skq")
        self.maskModel = project.version(1).model
        self.bridge = CvBridge()
        self.topic = "/usb_cam/image_raw"
        self.rate = rospy.Rate(5)
        self.img_sub = rospy.Subscriber(self.topic,imgmsg,self.camera_callback)
        time.sleep(1)
        rospy.Service('features', Features, self.handler)



    def camera_callback(self,data):
        self.cam_image = data


    def handler(self, request):
        self.recog = 0
        rospy.loginfo("Service called!")
        rospy.loginfo("Requested..")
        time.sleep(2)
        while self.recog == 0:
            self.img_sub = rospy.Subscriber(self.topic,imgmsg,self.camera_callback)
            out  = self.features()
            self.rate.sleep()
            return out
        cv2.destroyAllWindows()


    def features(self):
        frame = self.bridge.imgmsg_to_cv2(self.cam_image,desired_encoding='bgr8')
        time.sleep(1)
        
        cv2.imwrite('/home/bibo/catkin_fodase/src/Feature_extraction/features_pkg/src/data/frame.jpg', frame)    
        
        self.funCrop()
        pantscolor = self.funColor('/home/bibo/catkin_fodase/src/Feature_extraction/features_pkg/src/data/pants.jpg')
        shirtcolor = self.funColor('/home/bibo/catkin_fodase/src/Feature_extraction/features_pkg/src/data/shirt.jpg')
        glasses = self.ifGlasses('/home/bibo/catkin_fodase/src/Feature_extraction/features_pkg/src/data/head.jpg')
        mask = self.ifMask('/home/bibo/catkin_fodase/src/Feature_extraction/features_pkg/src/data/head.jpg')

        out= f'I really like your {pantscolor} pants and your {shirtcolor} shirt! I see youre {glasses} wearing glasses. And youre {mask} wearing a mask.'
        print(out)

        return out

    def funCrop(self):
        img = cv2.imread('/home/bibo/catkin_fodase/src/Feature_extraction/features_pkg/src/data/frame.jpg')
        results = self.personModel.predict(img)
        print(len(results))      

        result = results[0]  
        am = -100 
        for i, box in enumerate(result.boxes):
            if box.cls.item() == 0: 
                cords = box.xyxy[0].tolist()
                area = (cords[2] - cords[0]) * (cords[3] - cords[1])
                if area > am:
                    am = area
                    xyxy = cords
        crop = img[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]       
        #cv2.imwrite('/home/bibo/catkin_fodase/src/features/src/crop.jpg',crop)
        self.funKeypoints(crop)


    def zoom(self,img, zoom_factor=2):
        return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)


    def funKeypoints(self,img):
        results = self.keypointsModel(img) 
        result = results[0]
        result_keypoint = result.keypoints.xy.numpy()[0]
        ombro = int(result_keypoint[5][1])
        quadril = int(result_keypoint[11][1])
        nareba = int(result_keypoint[0][0])
        shirt = img[ombro:quadril,20:-20]
        pants = img[quadril:,:]
        height, width = pants.shape[:2] 
        pants = pants[:int(height/2),20:int(width/2)]
        head = img[:ombro,10:-10]
        head = self.zoom(head,3)
        cv2.imwrite('/home/bibo/catkin_fodase/src/Feature_extraction/features_pkg/src/data/head.jpg',head)
        cv2.imwrite('/home/bibo/catkin_fodase/src/Feature_extraction/features_pkg/src/data/pants.jpg',pants)
        cv2.imwrite('/home/bibo/catkin_fodase/src/Feature_extraction/features_pkg/src/data/shirt.jpg',shirt)
        '''
        cv2.circle(img,(nareba,ombro),2,(255,0,0),5)
        cv2.circle(img,(nareba,quadril),2,(0,255,0),5)
        cv2.imwrite('/home/bibo/catkin_fodase/src/features/src/keypoints.jpg',img)
        '''


    def funColor(self,path):
        color_thief = ColorThief(path)
        dominant_color = color_thief.get_color()
        R = dominant_color[0]
        G = dominant_color[1]
        B = dominant_color[2]

        if((R>=200 and R<=255)and(G>=200 and G<=255)and(B>=200 and G<=255)):
            cor = 'White'
        elif(R<=40 and G<=40 and B<=40):
            cor = 'Black'
        elif(R>=0 and R<=50):
            if(G>=0 or G<=50):
                cor = 'Blue'
            else:
                if (B>=0 and B<=50):
                    cor = 'Green'
                else: 
                    cor = 'Cyan'
        elif (R>=150 and R<=200):
            cor = 'Brown'
        elif (R>=0 and R<=150):
            if(G>=0 and G<=100):
                cor = 'Jeans'
            else: 
                if(B>=0 and B<=50):
                    cor = 'Brown'
                else: 
                    cor = 'Purple'
        elif(R>=200 and R<=255):
            if((G>=0 and G<=50) and (B>=0 and B<=50)):
                cor = 'Red'
            elif(G>=100 and G<=200):
                cor = 'Orange'
            elif(G>=200 and G<=255):
                if(B>=0 and B<=50):
                    cor = 'Yellow'
            
                else: 
                    cor = 'Pink'
        return cor
        

    def ifGlasses(self,path):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        img = dlib.load_rgb_image(path)
        rect = detector(img)[0]
        sp = predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        nose_bridge_x = []
        nose_bridge_y = []
        for i in [28,29,30,31,33,34,35]:
            nose_bridge_x.append(landmarks[i][0])
            nose_bridge_y.append(landmarks[i][1])

        ### x_min and x_max
        x_min = min(nose_bridge_x)
        x_max = max(nose_bridge_x)### ymin (from top eyebrow coordinate),  ymax
        y_min = landmarks[20][1]
        y_max = landmarks[31][1]
        img2 = Image.open(path)
        img2 = img2.crop((x_min,y_min,x_max,y_max))

        img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)

        #center strip
        edges_center = edges.T[(int(len(edges.T)/2))]

        if 255 in edges_center:
            return ' '
        else:
            return 'Not' 

    def ifMask(self,path):
        try:
            results = self.maskModel.predict(path, confidence=9, overlap=30).json()
            if results.get('predictions')[0].get('class') != 'Nomask':
                return ' '
            else:
                return 'Not'
        except:
            return 'Not' #Mais provável da pessoa estar sem máscara !!!!!!! TESTAR !!!!!!!

if __name__ == "__main__":
    rospy.init_node('feature_bonus', log_level=rospy.INFO)
    rospy.loginfo("Service started!")
    time.sleep(2)
    FeaturesRecognition()
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    
