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
import datetime
import sys
from hera_objects.msg import *
from hera_objects.srv import FindObject, FindSpecificObject

PATH = '/home/robofei/Workspace/catkin_ws/src/3rd_party/vision_system/Feature_extraction/features_pkg/src'
TOPIC = '/usb_cam/image_raw'
NAMED_COLORS = {
    'red': (255, 0, 0),
    'green': (0, 128, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'orange': (255, 140, 0),  # Adjusted orange
    'purple': (128, 0, 128),
    'brown': (139, 69, 19),  # Adjusted brown
    'white': (208, 206, 223),  # Adjusted white
    'whitee':(135, 135, 136),
    'black': (20, 20, 20),
    'blackk': (40, 40, 40),
    'gray': (128, 128, 128),  # Wider range of gray
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'lime': (0, 255, 0),
    'maroon': (128, 0, 0),
    'olive': (128, 128, 0),
    'navy': (5, 5, 40),  # Wider range of navy
    'gold': (255, 215, 0),
    'violet': (238, 130, 238),
    'turquoise': (64, 224, 208),  # Adjusted turquoise
    'beige': (245, 245, 220),
    'salmon': (250, 128, 114),
    'khaki': (240, 230, 140),
    'indigo': (75, 0, 130),  # Wider range of indigo
    'coral': (255, 127, 80),
    'tan': (210, 180, 140),
    'azure': (240, 255, 255),
    'sienna': (160, 82, 45),
    'crimson': (220, 20, 60)
}


class FeaturesRecognition:

    def __init__(self):
        self.personModel = YOLO(PATH + "/dep/yolov8m.pt")
        self.keypointsModel = YOLO(PATH + '/dep/yolov8n-pose.pt')
        self.maskModel = YOLO(PATH + '/dep/mask.pt')
        self.glassesModel = YOLO(PATH + '/dep/glasses.pt')
        self.bridge = CvBridge()
        self.topic = TOPIC
        self.rate = rospy.Rate(5)
        self.img_sub = rospy.Subscriber(self.topic,imgmsg,self.camera_callback)
        time.sleep(0.5)
        rospy.Service('features', Features, self.handler)
        self.objects = rospy.ServiceProxy('/objects', FindObject)
        self.pixel_height = 0


    def camera_callback(self,data):
        self.cam_image = data


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
            r, g, b = [int(x) for x in rgb]
            color_name = self.closest_color((r, g, b))
            return color_name
        except ValueError:
            return "Cor inválida"                


    def find_color(self,path):
        color_thief = ColorThief(path)
        dominant_color = color_thief.get_color()
        print(dominant_color)
        #cor = self.findColor(dominant_color)
        #cor = self.convert_rgb_to_names(dominant_color)
        cor = self.rgb_to_color_name(dominant_color)

        return cor


    def if_glasses(self,path):
        img = cv2.imread(path)
        results = self.glassesModel.predict(img, stream=True)     
        for result in results:                                        
            boxes = result.boxes.cpu().numpy()  
            print('box:',boxes.cls)

            try:
                box = boxes.cls[0]
                return ' indeed '                
            except:
                return ' not '


    def if_mask(self,path):
        img = cv2.imread(path)
        results = self.maskModel.predict(img, stream=True)                
        for result in results:                                        
            boxes = result.boxes.cpu().numpy()  
            if boxes.cls[0] != 0:
                return ' indeed '                
            else:
                return ' not '

       
    def crop_image(self):
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

        print("0:", xyxy[0])
        print("1:", xyxy[1])
        print("2:", xyxy[2])
        print("3:", xyxy[3])
        self.pixel_height = xyxy[1]
        crop = frame[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
        cv2.imwrite(PATH + '/data/crop.jpg', crop)
  
        self.find_keypoints(crop)


    def zoom(self,img, zoom_factor=2):
        return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)


    def find_keypoints(self,img):
        results = self.keypointsModel(img) 
        result = results[0]
        result_keypoint = result.keypoints.xy.cpu().numpy()[0]
        ombro = int(result_keypoint[5][1])
        quadril = int(result_keypoint[11][1])
        nareba = int(result_keypoint[0][0])      

        shirt = img[ombro+10:quadril-40,50:-50]
        pants = img[quadril+25:,:]
        height, width = pants.shape[:2] 
        pants = pants[:int(height/3),30:int(width/2)+10]
        head = img[:ombro,10:-10]
        head = self.zoom(head,3)
        cv2.imwrite(PATH +'/data/head.jpg',head)
        cv2.imwrite(PATH +'/data/pants.jpg',pants)
        cv2.imwrite(PATH +'/data/shirt.jpg',shirt)
        self.saturation(PATH + '/data/shirt.jpg')

    def find_closest_object(self):
        resp = self.objects("closest", "")
        coordinates = resp.position
        taken_object = resp.taken_object
        values = [coordinates[0].x, coordinates[0].y, coordinates[0].z]
        if all(v == 0.0 for v in values):
            return None, None
        else:
            return taken_object[0], coordinates[0]

    def height_estimate(self, height):
        obj_class, coords = self.find_closest_object()
        print(obj_class)
        distance = coords.x

        height = 1080 - height
        distance = (distance*100)+20
        camera_image_height = 1.48 * distance
        if height < 540:
            subject_height = 540 - height
            hf = 1.22 - (((subject_height*camera_image_height)/1080)/100)
        else:
            subject_height = height - 540
            hf = (((subject_height*camera_image_height)/1080)/100) + 1.22
        return hf

    def features(self):

        self.crop_image()
        pantscolor = self.find_color(PATH + '/data/pants.jpg')
        shirtcolor = self.find_color(PATH + '/data/shirt.jpg')
        glasses = self.if_glasses(PATH + '/data/head.jpg')
        mask = self.if_mask(PATH + '/data/head.jpg')
        hf = self.height_estimate(self.pixel_height)

        out= f'I really like your {pantscolor} pants and your {shirtcolor} top! I see youre{glasses}wearing glasses. And youre{mask}wearing a mask. You are between {(hf - 0.02):.2f} and {(hf + 0.02):.2f} meters tall. '

        current_time = datetime.datetime.now()

        with open(f"features log {current_time}.txt", "a") as myfile:
            myfile.write(out + "\n")

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
    