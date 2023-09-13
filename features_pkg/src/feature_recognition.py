#!/usr/bin/python3
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
import webcolors
from collections import namedtuple
import sys


#PATH = '/home/robofei/Workspace/catkin_ws/src/3rd_party/vision_system/Feature_extraction/features_pkg/src'
PATH = '/home/bibo/catkin_fodase/src/Feature_extraction/features_pkg/src'

NAMED_COLORS = {
    'red': (255, 0, 0),
    'green': (0, 128, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'pink': (255, 192, 203),
    'brown': (165, 42, 42),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (128, 128, 128),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'lime': (0, 255, 0),
    'teal': (0, 128, 128),
    'lavender': (230, 230, 250),
    'maroon': (128, 0, 0),
    'olive': (128, 128, 0),
    'navy': (0, 0, 40),
    'silver': (192, 192, 192),
    'gold': (255, 215, 0),
    'violet': (238, 130, 238),
    'turquoise': (64, 224, 208),
    'beige': (245, 245, 220),
    'salmon': (250, 128, 114),
    'khaki': (240, 230, 140),
    'indigo': (75, 0, 130),
    'coral': (255, 127, 80),
    'orchid': (218, 112, 214),
    'tan': (210, 180, 140),
    'azure': (240, 255, 255),
    'sienna': (160, 82, 45),
    'crimson': (220, 20, 60)
}



class FeaturesRecognition:

    def __init__(self):
        self.personModel = YOLO(PATH + "/dep/yolov8m.pt")
        self.keypointsModel = YOLO(PATH + '/dep/yolov8n-pose.pt')
        rf = Roboflow(api_key="NSkF2YE2ufLzrNUiswiA")
        project = rf.workspace().project("mask-detection-m3skq")
        self.maskModel = project.version(1).model
        self.bridge = CvBridge()
        #self.topic = "/camera/rgb/image_raw"
        self.topic = '/usb_cam/image_raw'
        self.rate = rospy.Rate(5)
        self.img_sub = rospy.Subscriber(self.topic,imgmsg,self.camera_callback)
        time.sleep(0.5)
        rospy.Service('features', Features, self.handler)


    def camera_callback(self,data):
        self.cam_image = data


    def funCrop(self):
        result = None
        while result is None:
            frame = self.bridge.imgmsg_to_cv2(self.cam_image,desired_encoding='bgr8')
            rospy.logwarn('Person not found... Trying again')
            results = self.personModel.predict(frame)[0]
            am = -100 
            for i, box in enumerate(results.boxes):
                if box.cls.item() == 0:
                    cords = box.xyxy[0].tolist()
                    area = (cords[2] - cords[0]) * (cords[3] - cords[1])
                    if area > am:
                        am = area
                        xyxy = cords
            if 'xyxy' in locals():
                result = results[0]

        rospy.loginfo('Frame set')
        cv2.imwrite(PATH + '/data/frame.jpg', frame)    

        crop = frame[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]       
        self.funKeypoints(crop)


    def zoom(self,img, zoom_factor=2):
        return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)


    def funKeypoints(self,img):
        results = self.keypointsModel(img) 
        result = results[0]
        result_keypoint = result.keypoints.xy.cpu().numpy()[0]
        ombro = int(result_keypoint[5][1])
        quadril = int(result_keypoint[11][1])
        nareba = int(result_keypoint[0][0])      

        shirt = img[ombro+10:quadril-40,30:-30]
        pants = img[quadril+25:,:]
        height, width = pants.shape[:2] 
        pants = pants[:int(height/3),30:int(width/2)+10]
        head = img[:ombro,10:-10]
        head = self.zoom(head,3)
        cv2.imwrite(PATH +'/data/head.jpg',head)
        cv2.imwrite(PATH +'/data/pants.jpg',pants)
        cv2.imwrite(PATH +'/data/shirt.jpg',shirt)
        self.saturation(PATH + '/data/shirt.jpg')


    def saturation(self,path):
        image = cv2.imread(path)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation_scale = 2.2
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_scale, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        cv2.imwrite(path, img)


    def color_distance(self, color1, color2):
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5


    def closest_color(self,requested_color):
        min_distance = sys.maxsize
        closest_name = None
        for name, color in NAMED_COLORS.items():
            distance = self.color_distance(requested_color, color)
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        return closest_name


    def rgb_to_color_name(self,rgb):
        try:
            # Normalizar os valores RGB para o intervalo [0, 255]
            r, g, b = [int(x) for x in rgb]
            # Encontrar a cor mais próxima na lista predefinida
            color_name = self.closest_color((r, g, b))
            return color_name
        except ValueError:
            return "Cor inválida"            
            


    def funColor(self,path):
        color_thief = ColorThief(path)
        dominant_color = color_thief.get_color()
        print(dominant_color)
        #cor = self.findColor(dominant_color)
        #cor = self.convert_rgb_to_names(dominant_color)
        cor = self.rgb_to_color_name(dominant_color)

        return cor
        

    def ifGlasses(self,path):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PATH + '/dep/shape_predictor_68_face_landmarks.dat')
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
        #cv2.imwrite('/home/bibo/catkin_fodase/src/Feature_extraction/features_pkg/src/data/test.jpg', edges)

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


    def features(self):

        self.funCrop()
        pantscolor = self.funColor(PATH + '/data/pants.jpg')
        shirtcolor = self.funColor(PATH + '/data/shirt.jpg')
        glasses = self.ifGlasses(PATH + '/data/head.jpg')
        mask = self.ifMask(PATH + '/data/head.jpg')

        out= f'I really like your {pantscolor} pants and your {shirtcolor} shirt! I see youre {glasses} wearing glasses. And youre {mask} wearing a mask.'

        return out


    def handler(self, request):
            self.recog = 0
            rospy.loginfo("Service called!")
            rospy.loginfo("Requested..")
            while self.recog == 0:
                self.img_sub = rospy.Subscriber(self.topic,imgmsg,self.camera_callback)
                out  = self.features()
                self.rate.sleep()
                return out
            cv2.destroyAllWindows()



if __name__ == "__main__":
    rospy.init_node('feature_bonus', log_level=rospy.INFO)
    rospy.loginfo("Service started!")
    FeaturesRecognition()
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    