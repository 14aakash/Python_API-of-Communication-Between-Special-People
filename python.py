import numpy
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import mediapipe as mp
from flask import Flask, request, jsonify
import pyodbc

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
# connection = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-4F8B9BO\\SQLEXPRESS;DATABASE=DD_TALK;Trusted_Connection=yes;')
server = 'AAKASH\AAKASH'
database = 'DD_TALK'
username = 'sa'
password = '15121472'
cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password)
cursor = cnxn.cursor()

# variables for dynamic hand signs
CLASSES_LIST = ["HELLO", "HOW ARE YOU", "I AM FINE", "THANK YOU", "WHERE ARE YOU FROM"]
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 10
LRCN_model = load_model("lib/Python/seq10.h5")
input_video_file_path = ''

# dynamic hand sign prediction code

def predict_single_action(video_file_path, SEQUENCE_LENGTH):
    ''',
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis=0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]

    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

    # Release the VideoCapture object.
    video_reader.release()
    return predicted_class_name

# setting parameters of image before prediction

def load_image(img_path, show=False):
    imgs = image.load_img(img_path, target_size=(700, 400))
    img_tensor = image.img_to_array(imgs)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()
    return img_tensor

# loading model for static hand sign gestures

model = load_model("lib/Python/Google_Everything.h5")
model2 = load_model("lib/Python/HOW_MN_WHEN.h5")

app = Flask(__name__)

@app.route('/static', methods=['POST'])
def get_detection():
    print("OKAY")
    file = request.files['video']
    fname = file.filename
    print("File Name is ", fname)
    print("File Type is ", file.content_type)
    file.save(os.path.join('lib/Python/Videos/', fname))
    print(file)
    i = 0
    predicted_signs = []
    previous_l = ''
    previous_l_count = 0
    present_l = ''
    word = ''
    dummy = ''
    query = ''
    next = 0
    ind = 0
    video = cv2.VideoCapture('lib/Python/Videos/' + fname)
    # video = cv2.VideoCapture('./Videos/TEST.mp4')
    # video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    startFrameNo = 0
    with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            startFrameNo += 14
            video.set(cv2.CAP_PROP_POS_FRAMES, startFrameNo)
            ret, imajes = video.read()
            if imajes is not None:
                imajes = cv2.resize(cv2.cvtColor(imajes, cv2.COLOR_RGB2BGR), (400, 700))
            elif imajes is None:
                break
            results = hands.process(imajes)
            imajes = cv2.resize(cv2.cvtColor(imajes, cv2.COLOR_RGB2BGR), (400, 700))
            s = cv2.imread('lib/Python/white400700.jpg')
            text = " "
            if results.multi_hand_landmarks:
                # for hand in results.multi_handedness:
                    # if hand.classification[0].index == 1:
                    #     print('Left')
                    # else:
                    #     print('Right')
                for hand_landmark in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(imajes, hand_landmark, mp_hand.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                    mp_draw.draw_landmarks(s, hand_landmark, mp_hand.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                    text = "Detected"
            cv2.imwrite(fname + str(i) + ".jpg", s)
            img_path = fname + str(i) + ".jpg"
            new_image = load_image(img_path)
            pred = model.predict(new_image)
            color = (98, 0, 238, 255)
            ahan = np.argmax(pred, axis=1)
            query = ' '.join(map(str, ahan))
            if text == "Detected":
                cursor.execute("Select * From VOCABULARY where DID=" + query)
                for row in cursor:
                    text = row[1]
            if text == 'M' or text == 'N' or text == 'HOW ' or text == 'WHEN ':
                minipred = model2.predict(new_image)
                miniahan = np.argmax(minipred, axis=1)
                if miniahan == [0]:
                    text = 'HOW '
                elif miniahan == [1]:
                    text = 'M'
                elif miniahan == [2]:
                    text = 'N'
                elif miniahan == [3]:
                    text = 'WHEN '
            print(text)
            predicted_signs.append(text)

            present_l = text
            if previous_l == '':
                previous_l = text
            elif present_l == previous_l:
                previous_l_count += 1
            elif present_l != previous_l:
                previous_l = present_l
                previous_l_count = 0
            if previous_l_count == 1 and ind > 0:
                if dummy != previous_l:
                    next = 1

            if previous_l_count == 2:
                if ind == 0:
                    word = previous_l
                    dummy = previous_l
                    previous_l_count = 0
                    previous_l = ''
                    ind = ind + 1
                elif dummy != previous_l:
                    word += previous_l
                    dummy = previous_l
                    previous_l_count = 0
                    previous_l = ''
                    ind = ind + 1
                    next = 0
                elif dummy == previous_l and next == 1:
                    word += previous_l
                    dummy = previous_l
                    previous_l_count = 0
                    previous_l = ''
                    ind = ind + 1
                    next = 0

            coordinates = (75, 75)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1.5
            thickness = 4
            imajes = cv2.putText(imajes, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow("Frame", imajes)
            cv2.imshow("Test", s)
            k = cv2.waitKey(1)
            os.remove(fname +str(i) + ".jpg")
            i += 1
            if k == ord('q'):
                break
    video.release()
    print("Predicted Signs List : ", predicted_signs)
    print("Predicted WORD : ", word)
    print("Total Frames = ", i)

    cv2.destroyAllWindows()
    os.remove("lib/Python/Videos/"+fname)
    return jsonify(word)
    # return jsonify({'word':word})


@app.route('/dynamic', methods=['POST'])
def dynamic():

    print("OKAY")
    file = request.files['video']
    fname = file.filename
    print("File Name is ", fname)
    print("File Type is ", file.content_type)
    file.save(os.path.join('lib/Python/Videos/', fname))
    print(file)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 30, (400, 700))
    i = 0

    video = cv2.VideoCapture('lib/Python/Videos/' + fname)
    # video = cv2.VideoCapture('./Videos/TEST.mp4')
    # video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    startFrameNo = 0
    with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            startFrameNo += 3
            video.set(cv2.CAP_PROP_POS_FRAMES, startFrameNo)
            ret, imajes = video.read()
            if imajes is not None:
                imajes = cv2.resize(cv2.cvtColor(imajes, cv2.COLOR_RGB2BGR), (400, 700))
            elif imajes is None:
                break
            results = hands.process(imajes)
            imajes = cv2.resize(cv2.cvtColor(imajes, cv2.COLOR_RGB2BGR), (400, 700))
            s = cv2.imread('lib/Python/white400700.jpg')
            text = " "
            if results.multi_hand_landmarks:
                # for hand in results.multi_handedness:
                # if hand.classification[0].index == 1:
                #     print('Left')
                # else:
                #     print('Right')
                for hand_landmark in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(imajes, hand_landmark, mp_hand.HAND_CONNECTIONS,
                                           mp_drawing_styles.get_default_hand_landmarks_style(),
                                           mp_drawing_styles.get_default_hand_connections_style())
                    mp_draw.draw_landmarks(s, hand_landmark, mp_hand.HAND_CONNECTIONS,
                                           mp_drawing_styles.get_default_hand_landmarks_style(),
                                           mp_drawing_styles.get_default_hand_connections_style())
                    text = "Detected"
            cv2.imwrite(fname +str(i) + ".jpg", s)
            img_path = fname + str(i) + ".jpg"
            out.write(s)
            cv2.imshow("Video", imajes)
            cv2.imshow("Landmarks", s)
            k = cv2.waitKey(1)
            os.remove(fname + str(i) + ".jpg")
            i += 1
            if k == ord('q'):
                break
    video.release()
    print("Total Frames = ", i)

    cv2.destroyAllWindows()
    os.remove("lib/Python/Videos/" + fname)
    out.release()
    input_video_file_path = 'output.mp4'
    # Perform Single Prediction on the Test Video.
    if i >= 10:
        answer = predict_single_action(input_video_file_path, SEQUENCE_LENGTH)
    else:
        answer = 'Please capture a video of at least 10 frames'
    print("Answer = ", answer)
    return jsonify(answer)


@app.route('/testing', methods=['GET'])
def Test():
    return 'Working'

if __name__ == '__main__':
    app.run(host='192.168.43.17', debug=True) #MY PHONE
    # app.run(host='127.0.0.1', debug=True)
    #app.run(debug=True)

