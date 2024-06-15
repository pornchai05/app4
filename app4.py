import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from yolo_predictions import YOLO_Pred

#-----------------------------------------------
import os
from twilio.rest import Client
account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)
token = client.tokens.create()
#-----------------------------------------------

yolo = YOLO_Pred('my_obj.onnx','my_obj.yaml') 

st.title("ตรวจจับวัตถุ : Webcam")

class VideoProcessor:  
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img,1)
        #--------------------------------------------------
        pred_image, obj_box = yolo.predictions(img)
        #--------------------------------------------------
        return av.VideoFrame.from_ndarray(pred_image,format="bgr24")


webrtc_streamer(key="test",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True,"audio": False},
                rtc_configuration={"iceServers":token.ice_servers}) ###


