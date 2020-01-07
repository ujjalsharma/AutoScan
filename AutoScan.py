from darkflow.net.build import TFNet
import tensorflow as tf
import numpy as np
import cv2
import imutils
from datetime import datetime

######################## PLATE DETECTION ################################################

# Function to return the cropped image of the detected plate from yolo json output
def detect_plate(input_image, pred_predictions_json):
    
    #sort the json using 
    pred_predictions_json.sort(key=lambda x: x.get('confidence'))

    #get the coordinates
    xtop = pred_predictions_json[-1].get('topleft').get('x')
    ytop = pred_predictions_json[-1].get('topleft').get('y')
    xbottom = pred_predictions_json[-1].get('bottomright').get('x')
    ybottom = pred_predictions_json[-1].get('bottomright').get('y')
    
    #crop the image using the coordinates
    output_image = input_image[ytop:ybottom, xtop:xbottom]
    
    cv2.rectangle(input_image,(xtop,ytop),(xbottom,ybottom),(0,255,0),3)
    return output_image

# Function to refine the cropped image of the detected plate in the main     
def refined_detect_plate(input_image):
    gray=cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,0)
    contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if(len(areas)!=0):
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        output_image = input_image[y:y+h,x:x+w]
    else: 
        output_image = input_image
    return output_image

########################################## CHARACTER SEGMENTATION #############################################

#Function to perform auto canny
def auto_canny(input_image, sigma=0.33):
    v = np.median(input_image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    output_image = cv2.Canny(input_image, lower, upper)
    return output_image

# Function to segment the letters of the cropped plate
def read_plate(input_image):
    
    character_list=[]
    
    gray = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,1)
    edges = auto_canny(thresh_inv)
    ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    img_area = input_image.shape[0]*input_image.shape[1]

    # go throgh each segment
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi_area = w*h
        non_max_sup = roi_area/img_area
        
        # identify and read the character by passing through the cnn classifier
        if((non_max_sup >= 0.015) and (non_max_sup < 0.09)):
            if ((h>1.2*w) and (3*w>=h)):
                char = input_image[y:y+h,x:x+w]
                # character classification
                character_list.append(character_recognition(char))
    licensePlate="".join(character_list)
    return licensePlate


#################################### CHARACTER RECOGNITION #############################################

# Function to predict a character in the license plate using the classifier
def character_recognition(input_image):
    
    # define the labels, same as while training the CNN model
    labels_dictionary = {0:'0', 1:'1', 2 :'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
    11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K',
    21:'L', 22:'M', 23:'N', 24:'P', 25:'Q', 26:'R', 27:'S', 28:'T', 29:'U',
    30:'V', 31:'W', 32:'X', 33:'Y', 34:'Z'}

    character_bw=cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    character_bw = cv2.resize(character_bw,(75,100))
    
    # image height, width and channel dimenstion same as while training the CNN model
    character_image = character_bw.reshape((1, 100,75, 1))
    character_image = character_image / 255.0
    cnn_output = char_classifier.predict(character_image)
    
    #Returning the character with highest probability
    predicted_character = np.argmax(cnn_output)
    
    return labels_dictionary[predicted_character]


# Load the trained yolo model and weights 
options = {"pbLoad": "load_files/yolo-plate.pb", "metaLoad": "load_files/yolo-plate.meta", "gpu": 0.9}
yoloPlate = TFNet(options)

# Load the cnn classifier
char_classifier = tf.keras.models.load_model('load_files/cnn_char_classifier_01.h5')

# Input video file path
input_video_path = 'video2.MOV'

#Capture frames from input video
video_frame_captured = cv2.VideoCapture(input_video_path)


num = 0
predictions_list = []
i=1

# loop through each frame
while(video_frame_captured.isOpened()):
    ret, frame = video_frame_captured.read()
    #frame = imutils.rotate(frame, 270)
    if num%6 == 0:
        try:
            # Plate Detection
            plate_detect_pred = yoloPlate.return_predict(frame) 
            detected_plate_image = detect_plate(frame, plate_detect_pred)
            refined_detected_plate_image = refined_detect_plate(detected_plate_image)
            
            # Plate Segmentation and Character recognition
            final_pred_plate = read_plate(refined_detected_plate_image)
            print("License Plate Prediction "+str(i) +str(" : ")+ str(final_pred_plate))
            i=i+1
            predictions_list.append(final_pred_plate)
        except:
            pass
    num+=1
    cv2.imshow('Input Video to Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_frame_captured.release()
cv2.destroyAllWindows()

# removing empty predictions and returning the maximum occuring prediction as final output
predictions_list = [x for x in predictions_list if x != '']
print("Final Prediction: "+ str(max(set(predictions_list), key=predictions_list.count)))


# Saving the output to a text file ReadPlateOutput.txt

text_file = open("ReadPlateOutput.txt", "a")
text_file.write("\n")
text_file.write("Date and Time of Recording: "+str(datetime.now())+"\n")
text_file.write("\n")
text_file.write("Video file : "+ str(input_video_path))
text_file.write("\n")
for i in range(len(predictions_list)):
    text_file.write("\n")
    text_file.write("Prediction "+str(i)+ str(" : ")+ str(predictions_list[i]))
    text_file.write("\n")

text_file.write("\n")
text_file.write("Final Prediction: "+ str(max(set(predictions_list), key=predictions_list.count)))
text_file.write("\n")
text_file.close()
