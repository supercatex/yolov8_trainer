#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import *


def callback_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
    

if __name__ == "__main__":
    rospy.init_node("test")
    rospy.loginfo("node start!")
    
    _image = None
    topic_image = "/camera/rgb/image_raw"
    rospy.Subscriber(topic_image, Image, callback_image)
    rospy.wait_for_message(topic_image, Image)
    
    net = Yolov8("best", "/home/pcms/runs/detect/custom/weights/best_openvino_model")
    net.classes = ['a']
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        
        frame = _image.copy()
        image = _image.copy()
        
        res = net.forward(image)[0]["det"]
        for x1, y1, x2, y2, score, label_id in res:
            if score < 0.9: continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
        