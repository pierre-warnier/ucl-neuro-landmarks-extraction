from re import template
import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
from validators import diners
from zipfile import ZipFile
import os
from datetime import datetime
import tempfile
import zipfile
import shutil
import copy

st.set_option('deprecation.showPyplotGlobalUse', False)

title = '<p style="font-family:Courier; color:Black; font-size: 60px; font-weight:bold;">Facial landmarks recognition</p>'
st.markdown(title, unsafe_allow_html=True)
st.text('Work hard... Play hard')

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]

image1 = cv2.imread('data/face.jpg')
rgb_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
result = face_mesh.process(rgb_image)    
height, width, _ = image1.shape
# Facial landmarks
result = face_mesh.process(image1)
#Black background
canvas = np.zeros((image1.shape[0], image1.shape[1], 1), dtype = "uint8")

def remove_index(i_drop_str, d_indexes):
    i_drop_l = i_drop_str.split(',')
    if i_drop_l != ['']:
        for i in i_drop_l:
            d_indexes.remove(int(i))
    return d_indexes
         

def dot_face(indexes=landmark_points_68, pxl=3):
    for facial_landmarks in result.multi_face_landmarks:
        for i in indexes :
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height) 
            cv2.circle(canvas, (x, y), pxl, (100, 100, 0), -1)            
    plt.imshow(canvas)
    fig = plt.show()
    st.pyplot(fig)

def video_processing(file, t_file, directory, indexes):
    cap = cv2.VideoCapture(t_file.name)
    ret, image2 = cap.read()
    height, width, channels = image2.shape
    # Naming file with current timestamp
    name = file.name
    out_path = f'{temp_dir}/{name}'
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (image2.shape[1], image2.shape[0]), isColor=False)
    while cap.isOpened():
        ret, image2 = cap.read()
        if ret is not True:
            break
        height, width, channels = image2.shape
        # Facial landmarks
        result = face_mesh.process(image2)
        # Creating black background
        canvas = np.zeros((image2.shape[0], image2.shape[1], 1), dtype = "uint8")
        if not result.multi_face_landmarks:
            continue            
        for facial_landmarks in result.multi_face_landmarks: 
            for i in indexes:
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                cv2.circle(canvas, (x, y), 3, (100, 100, 0), -1)
        out.write(canvas)
    cap.release()
    out.release()
    


st.subheader('Step 1: Upload video')
data = st.file_uploader("Upload a video", type=None, accept_multiple_files=True)

if len(data) >= 1 :
    with st.container():
        st.subheader('List of facial indexes')
        st.image('./data/image3.png')

    with st.container():
        st.subheader('Step 2: Set parameters and preview results')
        pixel = st.text_input('Pixel size as integer , eg: 3')
        i_drop_str = st.text_input('List of comma separated indexes, eg: 1,12,23,34,45...')

    with st.container():
        col1, col2, col3 = st.columns(3) 
        with col1:
            reset = st.button('Reset')
        with col2:        
            preview =  st.button('Preview')
        with col3:
            next = st.button('Next')

    with st.container():
        col1, col2 = st.columns(2) 
        upd_indices = landmark_points_68.copy()
        if reset:
            dot_face() 
        if preview:
            try:
                upd_indices = remove_index(i_drop_str, landmark_points_68)
            except ValueError as e:
                error = '<p style="font-family:Courier; color:Red; font-size: 20px; font-weight:bold;">Index not detected on face landmarks</p>'
                st.markdown(error, unsafe_allow_html=True)
            if len(pixel) >= 1:
                dot_face(upd_indices, int(pixel)) 
            else: 
                dot_face(upd_indices)
    with st.container():          
        if next:
            upd_indices = remove_index(i_drop_str, landmark_points_68)
            st.subheader('Step 3: Download your video... Enjoy!')
            dateTimeObj = datetime.now()
            dat = dateTimeObj.strftime("%Y-%m-%d-%H:%M:%S")
            with tempfile.TemporaryDirectory(prefix=dat) as temp_dir:
                for file in data:
                    with tempfile.NamedTemporaryFile() as t_file:
                        t_file.write(file.read())
                        video_processing(file, t_file, temp_dir, upd_indices)
                        #t_file.flush() 
                zip = shutil.make_archive('output_filename', 'zip', temp_dir)
                st.download_button(
                    label="Download videos as .zip",
                    data=zip,
                    file_name=f'analysis_',
                    mime='video/mp4')
                
                    
                    
    



            



