from re import template
import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
from zipfile import ZipFile
import os
from datetime import datetime
import tempfile
from zipfile import ZipFile
import shutil
import copy
import glob
import pytz


st.set_option('deprecation.showPyplotGlobalUse', False)

# Sanitize working directory
for f in glob.glob("20*"): # millenium bug
    try:
        os.remove(f)
    except:
        shutil.rmtree(f) 

title = '<p style="font-family:Courier; color:Black; font-size: 60px; font-weight:bold;">Facial landmarks recognition</p>'
st.markdown(title, unsafe_allow_html=True)
st.text('Work hard... Play hard')

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
#select sidebar
add_selectbox = st.sidebar.selectbox("Select model",("68 lanmarks", "468 lanmarks", "Cheat sheet"))
initial_iamge = './data/image3.png'
landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87] 
if add_selectbox == "468 lanmarks":   
    landmark_points_68 = list(range(0, 468))
    initial_iamge = './data/image2.png'

index_dict = {
    "Contours": [0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466], 
    "Oval Face": [132, 389, 136, 10, 397, 400, 148, 149, 150, 21, 152, 284, 288, 162, 297, 172, 176, 54, 58, 323, 67, 454, 332, 338, 93, 356, 103, 361, 234, 109, 365, 379, 377, 378, 251, 127], 
    "Eyes": [384, 385, 386, 387, 388, 390, 263, 362, 398, 466, 373, 374, 249, 380, 381, 382, 160, 33, 161, 163, 133, 7, 173, 144, 145, 246, 153, 154, 155, 157, 158, 159],
    "Lips": [0, 267, 269, 270, 13, 14, 17, 402, 146, 405, 409, 415, 291, 37, 39, 40, 178, 308, 181, 310, 311, 312, 185, 314, 317, 318, 61, 191, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375], 
    "Eye Brows": [293, 295, 296, 300, 334, 336, 276, 282, 283, 285, 65, 66, 70, 105, 107, 46, 52, 53, 55, 63], 
    "Irises": [474, 475, 476, 477, 472, 469, 470, 471], 
    "Left Eye": [384, 385, 386, 387, 388, 390, 263, 362, 398, 466, 373, 374, 249, 380, 381, 382], 
    "Left Eyebrow": [293, 295, 296, 300, 334, 336, 276, 282, 283, 285], 
    "Left Iris": [474, 475, 476, 477], 
    "Right Eye": [160, 33, 161, 163, 133, 7, 173, 144, 145, 246, 153, 154, 155, 157, 158, 159], 
    "Right Eyebrow": [65, 66, 70, 105, 107, 46, 52, 53, 55, 63], 
    "Right Iris": [472, 469, 470, 471]
    }


def remove_index(i_drop_str, d_indexes):
    i_drop_l = [int(i.strip()) for i in i_drop_str.split(',') if i != ''] 
    return set(d_indexes).difference(i_drop_l)

def dot_face(landmarks_list, canvas, width, height, indexes=landmark_points_68, pxl=3):
    for facial_landmarks in landmarks_list.multi_face_landmarks:
        for i in indexes :
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height) 
            cv2.circle(canvas, (x, y), pxl, (255,255,255), -1)
    return canvas 

def video_processing(file, t_file, directory, indexes, pixel_size):
    cap = cv2.VideoCapture(t_file.name)
    ret, image2 = cap.read()
    height, width, channels = image2.shape
    # Naming file with current timestamp
    name = file.name
    out_path = f'{directory}/{name}'
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (image2.shape[1], image2.shape[0]), isColor=False)
    while cap.isOpened():
        ret, image2 = cap.read()
        if ret is not True:
            break
        height, width, channels = image2.shape
        # Facial landmarks
        landmarks_list = face_mesh.process(image2)
        # Creating black background
        canvas = np.zeros((image2.shape[0], image2.shape[1], 1), dtype = "uint8")
        if not landmarks_list.multi_face_landmarks:
            continue            
        canvas = dot_face(landmarks_list, canvas, width, height, indexes=upd_indices, pxl=int(pixel_size))
        out.write(canvas)
    cap.release()
    out.release()
    return out_path

def image_processing(file, t_file, dest_directory, upd_indices, pixel_size):
    image = cv2.imread(t_file.name)
    height, width, _ = image.shape
    # Facial landmarks
    landmarks_list = face_mesh.process(image)
    #Black background
    canvas = np.zeros((image.shape[0], image.shape[1], 1), dtype = "uint8")
    canvas = dot_face(landmarks_list, canvas, width, height, indexes=upd_indices, pxl=pixel_size)
    cv2.imwrite(f'{dest_directory}/{file.name}', canvas)
    return f'{dest_directory}/{file.name}'

def image_or_video(s):
    ext = s.split('/')[-1].split('.')[-1]
    if ext in {'jpg', 'jpeg', 'png', 'bmp'}:
        return 'image'
    elif ext in {'mp4', 'm4v', 'webm', 'avi', 'mov'}:
        return 'video'

def download_callback(dest_path):
    shutil.rmtree(dest_path)
    os.remove(f"{dest_path}.zip")

st.subheader('Step 1: Upload video')
data = st.file_uploader("Upload a video", type=None, accept_multiple_files=True)

if add_selectbox ==  "Cheat sheet":
    selection = st.selectbox('Find indeces', ('Contours', 'Oval Face','Eyes','Lips','Eye Brows', 'Irises', 'Left Eye', 'Left Eyebrow','Left Iris','Right Eye', 'Right Eyebrow', 'Right Iris'))
    for key, value in index_dict.items():
        if selection == key:
            print(value)
            st.subheader(f'{value}')

if len(data) >= 1 :
    with st.container():
        st.subheader('List of facial indexes')
        st.image(initial_iamge)

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
            next = st.button('Process file')

    with st.container():
        try:
            pixel = int(pixel)
        except ValueError:
            pixel = 3

        col1, col2 = st.columns(2) 
        image1 = cv2.imread('data/face.jpg')
        rgb_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        height, width, _ = image1.shape
        # Facial landmarks
        landmarks_list = face_mesh.process(image1)
        #Black background
        canvas = np.zeros((image1.shape[0], image1.shape[1], 1), dtype = "uint8")
        if reset:
            canvas = dot_face(landmarks_list, canvas, width, height) 
            plt.imshow(canvas) # To investigate
            fig = plt.show()
            st.pyplot(fig)
        if preview:
            try:
                upd_indices = remove_index(i_drop_str, landmark_points_68)
            except ValueError as e:
                error = '<p style="font-family:Courier; color:Red; font-size: 20px; font-weight:bold;">Index not detected on face landmarks</p>'
                st.markdown(error, unsafe_allow_html=True)
            
            canvas = dot_face(landmarks_list, canvas, width, height, indexes=upd_indices, pxl=pixel) 
            plt.imshow(canvas) # To investigate
            fig = plt.show()
            st.pyplot(fig)

    with st.container():          
        if next:
            upd_indices = remove_index(i_drop_str, landmark_points_68)
            st.subheader('Step 3: Download your video... Enjoy!')
            timezone = pytz.timezone('Europe/Brussels')
            dateTimeObj = datetime.now(timezone)
            out_files = []
            dest_path = dateTimeObj.strftime("%Y%m%d_%H%M%S")            
            os.mkdir(dest_path)
            for file in data:
                with tempfile.NamedTemporaryFile() as t_file:
                    t_file.write(file.read())
                    out_path = None
                    file_type = image_or_video(file.name)
                    if file_type == 'image':
                        out_path = image_processing(file, t_file, dest_path, upd_indices, pixel_size=pixel)
                    elif file_type == 'video':
                        out_path = video_processing(file, t_file, dest_path, upd_indices, pixel_size=pixel)
                    if out_path is not None:
                        out_files.append(out_path)
                    #t_file.flush() 
                    # writing files to a zipfile

            with ZipFile(f'{dest_path}.zip','w') as zip:
                # writing each file one by one
                for file in out_files:
                    zip.write(file)
            with open(f'{dest_path}.zip', 'rb') as zip:
                st.download_button(
                    label="Download files as .zip",
                    file_name=f'{dest_path}.zip',
                    data=zip,
                    on_click=download_callback,
                    args=(dest_path,),
                    mime='application/zip')
        
                
                




        



