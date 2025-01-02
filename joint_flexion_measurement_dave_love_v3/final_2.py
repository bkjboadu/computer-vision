
import tensorflow as tf
import cv2
from pathlib import Path
import os,uuid
import mediapipe as mp
from tensorflow.keras.models import load_model
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import numpy as np,pandas as pd

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PyPDF2 import PdfMerger

import os,cvzone
from ultralytics import YOLO
from sort import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input',help='input file path',required=True)
args = parser.parse_args()

input_filename = args.input

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def get_angles(results):
        mp_pose = mp.solutions.pose
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

        # Convert landmarks to numpy arrays
        wrist = np.array([wrist.x, wrist.y, wrist.z])
        index = np.array([index.x, index.y, index.z])
        elbow = np.array([elbow.x, elbow.y, elbow.z])

        # Calculate the vectors
        v1 = index - wrist
        v2 = elbow - wrist

        # Calculate the angle using arccos
        wrist_angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

        wrist_angle = wrist_angle.astype(int)


        # Calculate the right elbow angle
        shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        shoulder = np.array([shoulder.x, shoulder.y])
        elbow = np.array([elbow.x, elbow.y])
        wrist = np.array([wrist.x, wrist.y])
        v1 = shoulder - elbow
        v2 = wrist - elbow
        elbow_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        elbow_angle = np.degrees(elbow_angle).astype(int)


        # Calculate the right shoulder angle
        hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        shoulder = np.array([shoulder.x, shoulder.y,shoulder.z])
        hip = np.array([hip.x, hip.y,hip.z])
        elbow = np.array([elbow.x, elbow.y,elbow.z])

        v1 = elbow[:2] - shoulder[:2]
        v2 = hip[:2] - shoulder[:2]
        shoulder_angle = np.arccos(np.dot(v2, v1) / (np.linalg.norm(v2) * np.linalg.norm(v1)))
        shoulder_angle = np.degrees(shoulder_angle).astype(int)
        
        if elbow[0] < shoulder[0]:
            shoulder_angle = -shoulder_angle
        else:
            if abs(shoulder_angle) >= 60:
                v1 = elbow - shoulder
                v2 = hip - shoulder
                shoulder_angle = np.arccos(np.dot(v2, v1) / (np.linalg.norm(v2) * np.linalg.norm(v1)))
                shoulder_angle = np.degrees(shoulder_angle).astype(int)
            else:
                shoulder_angle = abs(shoulder_angle)

        # Calculate the hip angle
        hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        hip = np.array([hip.x, hip.y,hip.z])
        knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        knee = np.array([knee.x, knee.y, knee.z])
        shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder = np.array([shoulder.x, shoulder.y,shoulder.z])
        v1 = shoulder - hip
        v2 = knee - hip
        hip_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        hip_angle = np.degrees(hip_angle).astype(int)

        # Calculate the right knee angle
        ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        ankle = np.array([ankle.x, ankle.y, ankle.z])
        v1 = hip - knee
        v2 = ankle - knee
        knee_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        knee_angle = np.degrees(knee_angle).astype(int)

        # Calculate the right ankle angle
        heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
        heel = np.array([heel.x, heel.y, heel.z])
        v1 = knee - ankle
        v2 = heel - ankle
        ankle_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        ankle_angle = np.degrees(ankle_angle).astype(int) 

        return [wrist_angle,elbow_angle,shoulder_angle,hip_angle,knee_angle,ankle_angle]



def display_angle_table(frame,results,width,height):
    
    wrist_angle,elbow_angle,shoulder_angle,hip_angle,knee_angle,ankle_angle = get_angles(results)

    cv2.rectangle(frame, (width - 600, height - 250), (width, height), (0, 0, 0), cv2.FILLED)
    cv2.line(frame,(width - 600, height - 250), (width - 600, height ), (255,255,255), 4)
    for i in [250,210,170,130,90,50]:
        cv2.line(frame, (width - 600, height - i), (width, height - i), (255,255,255), 4)
    cv2.putText(frame, f'wrist angle: {wrist_angle}', (width - 600, height - 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'elbow angle: {elbow_angle}', (width - 600, height - 180), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'shoulder angle: {shoulder_angle}', (width - 600, height - 140), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Hip angle: {hip_angle}', (width - 600, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'knee angle: {knee_angle}', (width - 600, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'ankle angle: {ankle_angle}', (width - 600, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    return [wrist_angle,elbow_angle,shoulder_angle,hip_angle,knee_angle,ankle_angle]


def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return results

def draw_landmarks(image,results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(10, 250, 80), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(250, 0, 0), thickness=2, circle_radius=2),)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 250, 80), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(250, 0, 250), thickness=2, circle_radius=2),)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(10, 250, 80), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 121, 255), thickness=2, circle_radius=2),)
    

def extract_keypoint(results):
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh])

def extract_images(video_path,output_folder,frames_to_extract):
    '''
    extracts images from video
    
    Arguments: 
        video_path: this is the path to the video we are extracting image from
        output_file: this is the directory our extracted image will be saved to
        frame_interval_seconds: this is the seconds interval to save each image
    '''

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_on_save = 0

    Path(output_folder).mkdir(exist_ok=True)
    folder_name = Path(video_path).stem

    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Save the frame if it's within the desired interval
        if  frame_count in frames_to_extract:

            final_npy_dir = Path(output_folder,folder_name,f"{frame_on_save}.npy")
            image_np = np.array(frame)
        
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)
            
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            # for i in range(len(detections['detection_boxes'])):
            box = detections['detection_boxes'][0]
            ymin, xmin, ymax, xmax = box
            ymin, xmin, ymax, xmax = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)

            player_frame = image_np[ymin:ymax, xmin:xmax]
            output_path = Path(output_folder, f"{str(uuid.uuid1())}.jpg")
            print(output_path)
            cv2.imwrite(str(output_path), player_frame)
    cap.release()

def view_fn(path,starting_point):
    cap = cv2.VideoCapture(path)
    frame_count = 0
    frame_to_save = 0
    folder_name = Path(path).stem
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            image_np = np.array(frame)
            
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)
            
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            # for i in range(len(detections['detection_boxes'])):
            box = detections['detection_boxes'][0]
            ymin, xmin, ymax, xmax = box
            ymin, xmin, ymax, xmax = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)

            player_frame = image_np[ymin:ymax, xmin:xmax]
            
            margin = 50

            text_x = max(min(xmin - margin, player_frame.shape[0] - 150), 0)
            text_y = max(min(ymin - margin, player_frame.shape[1] - 150), 0)

            
            if text_x < player_frame.shape[1] and text_y < player_frame.shape[0]:
                cv2.putText(player_frame, f'{frame_count}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                print("Text coordinates exceed frame dimensions.")
            cv2.imshow('frame',player_frame)
            

            if frame_count in np.arange(starting_point,starting_point + 25):
                image,results = mediapipe_detection(player_frame,holistic)
                extracted_results = extract_keypoint(results)
                final_npy_dir = Path('shot_detection_images','0',folder_name,f"{frame_to_save}.npy")
                Path('shot_detection_images','0',folder_name).mkdir(exist_ok=True)
                np.save(final_npy_dir,extracted_results)
                draw_landmarks(image,results)
                mp_drawing.draw_landmarks(player_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(10, 250, 80), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 0, 0), thickness=2, circle_radius=2),)
                frame_to_save += 1
            elif frame_count > starting_point + 25:
                break

        cap.release()
        cv2.destroyAllWindows()

def create_dataframe(keypoints,name_to_save):
    df = pd.DataFrame(keypoints,columns=['wrist_angle','elbow_angle','shoulder_angle','hip_angle','knee_angle','ankle_angle'])
    df.index = range(1, len(df) + 1)
    df.to_excel(f"{name_to_save}.xlsx")

def kalman_filter(y):
    y = np.array(y)
        # Initialize Kalman filter parameters
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.F = np.array([[1]])
    kf.H = np.array([[1]])
    # Process noise covariance
    kf.Q = np.array([[1e-3]])
    # Measurement noise covariance
    kf.R = np.array([[1e-1]])
    # Initial state
    kf.x = np.array([y[0]])

    # Initial state covariance
    kf.P *= 1e-2
    # Apply the Kalman filter to smooth the noisy data
    smoothed_values = []
    for measurement in y:
        kf.predict()
        kf.update(measurement)
        smoothed_values.append(kf.x[0])
    return smoothed_values


def plot_angle_graph_and_save(frames,angles,save_as,release_frame):
    x = np.array(frames)
    y = np.array(angles)

    def normalize_to_range(arrays, target_range=(-10, 30)):
        # Find the min and max values across all arrays
        min_val = np.min(arrays)
        max_val = np.max(arrays)

        # Calculate the scaling factor
        scale_factor = (target_range[1] - target_range[0]) / (max_val - min_val)

        # Normalize each array
        normalized_arrays = [(array - min_val) * scale_factor + target_range[0] for array in arrays]

        return normalized_arrays

    x = x[:len(y)]
    labels = ["wrist_angle", "elbow_angle", "shoulder_angle", "hip_angle", "knee_angle", "ankle_angle"]
    labels_to_normalize = ["elbow_angle","hip_angle", "knee_angle", "ankle_angle"]
    normalization_threshold = { "hip_angle":(-30,0), "knee_angle":(0,-45), "ankle_angle":(-20,40)}
    df = pd.DataFrame(y,index=x,columns=labels)
    # df.to_excel(f'final_{str(x[0])}.xlsx')
    


    labels_to_plot = ["elbow_angle", "shoulder_angle", "wrist_angle","hip_angle", "knee_angle", "ankle_angle"]

    for col in labels_to_plot:
        y = df.loc[:,col]
        

        if col in labels_to_plot[3:]:
            y = normalize_to_range(y)

        if col != 'wrist_angle':
            y_smooth = kalman_filter(y)
        else:
            y = [-value for value in y]
            y_smooth = kalman_filter(y)
            
        if col in labels_to_plot[3:]:
            plt.plot(x, y_smooth, label=col,linewidth=0.4)
        else:
            plt.plot(x, y_smooth, label=col)

    plt.axvline(x=release_frame, color='red', linestyle='--', label='Released')
    plt.title("Joint Flexion graph")
    plt.xlabel("frames")
    plt.ylabel("angles")
    plt.legend(loc="upper left", prop={'size':5})

    # Save the plot
    plt.savefig(save_as)
    plt.close()


def resize_image(img):
    target_width = 100
    height, width, _ = img.shape
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    resized_img = cv2.resize(img, (target_width, target_height))

    return resized_img


def create_pdf(pdf_filename, image_paths, folder_name,angles):
    c = canvas.Canvas(pdf_filename, pagesize=letter)

    # Add a header with the folder name
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, f'Shot: #{folder_name}')


    # Draw the first image on the left side
    c.drawImage(image_paths[0], 100, 500, width=80, height=100)

    # Draw the second image on the right side
    c.drawImage(image_paths[1], 200, 500, width=300, height=200)
    c.showPage()
    c.save()

def create_combined_pdf(main_folder, pdf_filename,angles):
    merger = PdfMerger()
    print(angles)
    # Iterate through subfolders
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)

        # Check if the item in the main folder is a subfolder
        if os.path.isdir(subfolder_path):

            # List image files in the subfolder
            image_files = [file for file in os.listdir(subfolder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Check if there are exactly 2 image files
            i = 0
            if len(image_files) == 2:
                image_paths = [os.path.join(subfolder_path, file) for file in image_files]
                
                # sheet_filename = f'{subfolder}.pdf'
                sheet_filename = os.path.join(subfolder_path,f'{subfolder}.pdf')

                # Create a PDF for each subfolder
                create_pdf(sheet_filename, image_paths, subfolder,angles[i])
                print(f'PDF created for {subfolder}')

                # Append the sheet to the main PDF
                merger.append(sheet_filename)
                i += 1
            else:
                print(f'Subfolder {subfolder} does not contain 2 images.')

    # Save the final PDF
    merger.write(pdf_filename)
    merger.close()

    # Remove individual subfolder PDFs
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            pdf_path = os.path.join(main_folder,subfolder_path, f'{subfolder}.pdf')
            print(pdf_path)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    print('PDF creation complete.')


def get_shot_pose_image(frame_num, player_angles, player_frame):
    assert len(frame_num) == len(player_angles) == len(player_frame)
    cols = ['wrist_angle', 'elbow_angle', 'shoulder_angle', 'hip_angle', 'knee_angle', 'ankle_angle']
    df = pd.DataFrame(player_angles, index=np.arange(len(frame_num)).tolist(), columns=cols)

    # Get the indices of the first 5 maximum values of shoulder_angle
    shoulder_max_indices = df['shoulder_angle'].nlargest(10).index.tolist()

    # Choose the first 3 maximum values of elbow_angle within the shoulder_max_indices
    # wrist_max_indices = df.loc[shoulder_max_indices]['wrist_angle'].nsmallest(3).index.tolist()
    elbow_max_indices = df.loc[shoulder_max_indices]['elbow_angle'].nsmallest(7).index.tolist()

    # Choose the index with the lowest value of wrist_angle within the elbow_max_indices
    final_index = df.loc[elbow_max_indices]['wrist_angle'].idxmin()
    # final_index = df.loc[wrist_max_indices]['elbow_angle'].idxmin()

    return final_index

def get_contact_prediction_and_index(images):
    results = []
    model = load_model('contact_detection.h5')

    def preprocess(img):
        handball_frame = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        handball_contact = tf.image.resize(handball_frame,(120,120))
        normalized_contact = handball_contact / 255.0
        handball_contact = np.expand_dims(normalized_contact,axis=0)
        prediction = model.predict(handball_contact)
        return prediction[0]
        
    for image in images:
        results.append(preprocess(image))
    
    index = np.argmax(results)
    return index
        

coco_classes = [
    "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck", "Boat", "Traffic light",
    "Fire hydrant", "Stop sign", "Parking meter", "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow",
    "Elephant", "Bear", "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee",
    "Skis", "Snowboard", "Sports ball", "Kite", "Baseball bat", "Baseball glove", "Skateboard", "Surfboard",
    "Tennis racket", "Bottle", "Wine glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple",
    "Sandwich", "Orange", "Broccoli", "Carrot", "Hot dog", "Pizza", "Donut", "Cake", "Chair", "Couch",
    "Potted plant", "Bed", "Dining table", "Toilet", "TV", "Laptop", "Mouse", "Remote", "Keyboard",
    "Cell phone", "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase",
    "Scissors", "Teddy bear", "Hair drier", "Toothbrush"
]

def init_video_capture_and_writer(path):
    ''' initialize video capture and writer'''
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = 'output_vid/' + str(Path(path).stem) + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_writer = cv2.VideoWriter(output_video_path,fourcc,fps,(width,height))
    return cap, fps, output_video_writer, width, height

def process_frame(frame,model_yolo,labels):
    detections = np.empty((0,5))
    results = model_yolo(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1,y1,x2,y2 = np.array(box.xyxy[0]).astype(int)
            conf = box.conf[0]
            cls = labels[int(box.cls[0])]
            if cls == 'Person' and conf>0.3:
                currentArray = [x1,y1,x2,y2,conf]
                detections = np.vstack((detections,currentArray))
    return detections

def analyze_pose_and_predict(sequence,y_predict, actions):
    global shot_buffer,waiting_buffer,shot_frame_count,WAITING_BUFFER_SIZE
    y_predict = np.argmax(y_predict)
    shot_label = actions[y_predict]
    shot_buffer.append(shot_label)
    waiting_buffer.append(shot_label)

    if len(shot_buffer) > BUFFER_SIZE:
        shot_buffer.pop(0)

    if len(waiting_buffer) > WAITING_BUFFER_SIZE:
        waiting_buffer.pop(0)

    if shot_buffer.count('shot') >= THRESHOLD:
        shot_frame_count += 1
    
    


def handle_shot_detection(shot_frame_count_max,WAITING_BUFFER_SIZE,shot_buffer,BUFFER_SIZE,shot_frame_count, waiting_buffer, player_image_frame, at_shot_frame_num, at_shot_frame_angles, shot_num, path):
    global angles_to_pdf
    
    if shot_frame_count >= shot_frame_count_max:
        if waiting_buffer[-WAITING_BUFFER_SIZE:].count('noshot') == round(0.7*WAITING_BUFFER_SIZE):
            shot_num += 1
            max_contact_index = get_contact_prediction_and_index(player_image_frame)
            
            frames_to_plot = at_shot_frame_num[max_contact_index-70:max_contact_index+130]
            angles_to_plot = at_shot_frame_angles[max_contact_index-70:max_contact_index+130]
            angles_for_pdf.append(at_shot_frame_angles[max_contact_index])

            Path('report').mkdir(exist_ok=True)
            at_shot_dir = Path('report') / Path(path).stem / f"{shot_num}"
            at_shot_dir.mkdir(parents=True,exist_ok=True)
            img_path = at_shot_dir / f"{shot_num}.jpg"
            graph_path = at_shot_dir / f"graph_{shot_num}.png"
            shot_frame_count = 0

            plot_angle_graph_and_save(frames_to_plot,angles_to_plot,str(graph_path),release_frame=at_shot_frame_num[max_contact_index])
            cv2.imwrite(str(img_path),player_image_frame[max_contact_index])
            at_shot_frame_num = []
            at_shot_frame_angles = []
            player_image_frame = []
            shot_frame_count = 0
        else:
            if shot_buffer[-BUFFER_SIZE:].count('noshot') == BUFFER_SIZE:
                shot_frame_count = 0
    
    return shot_num, at_shot_frame_num, at_shot_frame_angles, player_image_frame, shot_frame_count

yolo_model = YOLO('yolov8l.pt')

def do_analysis(path):    
    player_image_frame = []
    player_image_keypoint = []
    at_shot_frame_num = []
    at_shot_frame_angles = []
    actions = {0:'noshot',1:'shot'}

    tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
    first_frame = True
    frame_count = 0
    cap, fps, output_video_writer, width, height= init_video_capture_and_writer(path)

    BUFFER_SIZE = 20 if fps < 70 else 30  # Adjust as needed
    WAITING_BUFFER_SIZE = BUFFER_SIZE + 20

    THRESHOLD = 10 if fps < 70 else 25 # Adjust as needed
    shot_frame_count_max = 12 if fps < 70 else 8
    shot_buffer = []
    waiting_buffer = []
    shot_frame_count= 0
    sequences = []
    mp_pose = mp.solutions.pose
    shot_num = 0
    angles_for_pdf = []
    total_count = []
    labels = ['Person']

    shot_model = load_model('shot_detection_lstm.h5')
    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
               print('done')
               break

            if not ret:
               break

            frame_count += 1
            at_shot_frame_num.append(frame_count)
            
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Shot number: {shot_num}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(image_np, f'Shot frame count: {shot_frame_count}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


            detections = process_frame(frame,yolo_model,coco_classes)
            
            track_results = tracker.update(detections)
            for result in track_results:
                x1,y1,x2,y2,id = np.array(result).astype(int)
                if first_frame:
                    w,h = x2-x1,y2-y1

                    cv2.rectangle(frame, (max(0,x1-70), max(0,y1-70)), (max(0,x2+70), max(0,y2+70)), (0,255,0), 2)
                    cvzone.putTextRect(frame,str(id),(max(0,x1-70),max(20,y1-70)),scale=1,thickness=2,offset=5)
                elif id == object_id_to_track:
                    w,h = x2-x1,y2-y1

                    cv2.rectangle(frame, (max(0,x1-70), max(0,y1-70)), (max(0,x2+70), max(0,y2+70)), (0,255,0), 2)
                    cvzone.putTextRect(frame,str(id),(max(0,x1-70),max(20,y1-70)),scale=1,thickness=2,offset=5)     

                    if w>0 and h>0:
                        player_frame = frame[max(0,y1-70):max(0,y2+70), max(0,x1-70):max(0,x2+70)]

                    if player_frame.size != 0:
                            player_frame_to_save = player_frame.copy()
                            player_image_frame.append(player_frame_to_save)
                            results = mediapipe_detection(player_frame,holistic)
                            keypoints = extract_keypoint(results)
                            player_image_keypoint.append(keypoints)

                    draw_landmarks(player_frame,results)
                    if results.pose_landmarks:
                        wrist_angle,elbow_angle,shoulder_angle,hip_angle,knee_angle,ankle_angle = display_angle_table(frame,results,width,height)
                        item_list = [wrist_angle,elbow_angle,shoulder_angle,hip_angle,knee_angle,ankle_angle]
                        at_shot_frame_angles.append(item_list)
                    else:
                        item_list = [0,0,0,0,0,0]
                        at_shot_frame_angles.append(item_list)
                    
                    sequences.append(keypoints)
                    sequence = sequences[-5:]
                    if len(sequence) == 5:
                        y_predict = shot_model.predict(np.expand_dims(sequence,axis=0))
                        y_predict = np.argmax(y_predict)
                        shot_label = actions[y_predict]
                        shot_buffer.append(shot_label)
                        waiting_buffer.append(shot_label)

                        if len(shot_buffer) > BUFFER_SIZE:
                           shot_buffer.pop(0)

                        if len(waiting_buffer) > WAITING_BUFFER_SIZE:
                           waiting_buffer.pop(0)

                        if shot_buffer.count('shot') >= THRESHOLD // 2:
                           shot_frame_count += 1

                        if shot_frame_count >= shot_frame_count_max:
                           if waiting_buffer[-WAITING_BUFFER_SIZE:].count('noshot') == round(0.7*WAITING_BUFFER_SIZE):
                              shot_num += 1
                              max_contact_index = get_contact_prediction_and_index(player_image_frame)
                              
                              frames_to_plot = at_shot_frame_num[max_contact_index-70:max_contact_index+130]
                              angles_to_plot = at_shot_frame_angles[max_contact_index-70:max_contact_index+130]
                              angles_for_pdf.append(at_shot_frame_angles[max_contact_index])

                              Path('report').mkdir(exist_ok=True)
                              at_shot_dir = Path('report') / Path(path).stem / f"{shot_num}"
                              at_shot_dir.mkdir(parents=True,exist_ok=True)
                              img_path = at_shot_dir / f"{shot_num}.jpg"
                              graph_path = at_shot_dir / f"graph_{shot_num}.png"
                              shot_frame_count = 0

                              plot_angle_graph_and_save(frames_to_plot,angles_to_plot,str(graph_path),release_frame=at_shot_frame_num[max_contact_index])
                              cv2.imwrite(str(img_path),player_image_frame[max_contact_index])
                              at_shot_frame_num = []
                              at_shot_frame_angles = []
                              player_image_frame = []
                              shot_frame_count = 0
                        else:
                           if shot_buffer[-BUFFER_SIZE:].count('noshot') == BUFFER_SIZE:
                              shot_frame_count = 0
                    
            if first_frame:
                cv2.imshow('frame', cv2.resize(frame, (1200, 600)))
                cv2.waitKey(0)
                object_id_to_track = int(input('Enter the object id you want to track:'))
                first_frame = False
            else:
                cv2.imshow('frame', cv2.resize(frame, (1200, 600)))
                output_video_writer.write(frame)

        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    output_video_writer.release()
            
    pdf_init_path = os.path.join('report',f"{str(Path(path).stem)}")
    Path(pdf_init_path).mkdir(parents=True,exist_ok=True)
    create_combined_pdf(pdf_init_path, os.path.join(pdf_init_path,f"{str(Path(path).stem)}_shot_report.pdf"),angles_for_pdf)


if __name__ == '__main__':
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f'The file path "{input_filename}" does not exist')
    else:
        try:
            do_analysis(input_filename)
        except PermissionError:
            raise PermissionError("a file for this report is already opened in another app. Please close and run code again")





