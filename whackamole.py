# import os
# os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # 실행시 오래걸리는 컴퓨터에선 주석 해제
import cv2
import mediapipe as mp

import random

import time

import math

import pygame

pygame.init()
whack_sound = pygame.mixer.Sound('boing.wav')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



# 투명한 영역이 있는 이미지 영상에 오버레이하는 함수
def overlay(image, x, y, w, h, overlay_image): # 대상 이미지 (3채널), x, y 좌표, width, height, 덮어씌울 이미지 (4채널)
    alpha = overlay_image[:, :, 3] # BGRA
    mask_image = alpha / 255 # 0 ~ 255 -> 255 로 나누면 0 ~ 1 사이의 값 (1: 불투명, 0: 완전)
    # (255, 255)  ->  (1, 1)
    # (255, 0)        (1, 0)
    
    # 1 - mask_image ?
    # (0, 0)
    # (0, 1)
    
    for c in range(0, 3): # channel BGR
        image[y-h:y+h, x-w:x+w, c] = (overlay_image[:, :, c] * mask_image) + (image[y-h:y+h, x-w:x+w, c] * (1 - mask_image))




# 두 포인트 사이 거리 구하는 함수
def get_distance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance



game_start_event = False
game_over_event = False
game_pause_event = False




time_given=30.9
time_remaining = 99



score = 0




# rect_color0 = (0, 128, 255) 
# rect_color1 = (0, 255, 255) 

rx0=random.randint(50, 590)
ry0=random.randint(50, 430)
# rx0=random.randint(0, 540)
# ry0=random.randint(0, 380)
r0 = []
r0.append(rx0)
r0.append(ry0)


rx1=random.randint(50, 590)
ry1=random.randint(50, 430)
# rx1=random.randint(0,540)
# ry1=random.randint(0,380)
r1 = []
r1.append(rx1)
r1.append(ry1)


# 이미지 선언
mole_image = cv2.imread('mole_tr100.png', cv2.IMREAD_UNCHANGED)
moleh, molew, _ = mole_image.shape

shine_image = cv2.imread('shine.png', cv2.IMREAD_UNCHANGED)
shineh, shinew, _ = shine_image.shape

clap_image = cv2.imread('clap.png', cv2.IMREAD_UNCHANGED)
claph, clapw, _ = clap_image.shape


# For webcam input:
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():

        success, frame = cap.read()

        image = cv2.flip(frame, 1)

        


        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Make detection하는 부분
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        # Recolor back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape        

        present_time = time.time()





        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            rightindex = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            leftindex = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]       

            righthand = [rightindex[0]*w, rightindex[1]*h]
            lefthand = [leftindex[0]*w, leftindex[1]*h]            
            # print(int(rightindex[0]*10), int(rightindex[1]*10),int(rightindex[2]*10))
            

            noseindex = [landmarks[0].x, landmarks[0].y]
            nose = [noseindex[0]*w, noseindex[1]*h]

            rightfootindex = [landmarks[31].x, landmarks[31].y]
            leftfootindex = [landmarks[32].x, landmarks[32].y]
            rightfoot = [rightfootindex[0]*w, rightfootindex[1]*h]
            leftfoot = [leftfootindex[0]*w, leftfootindex[1]*h]




            if game_start_event == False:
                cv2.ellipse(image, (w//2, h//2-50), (72, 90) ,0 ,0, 360, (0,0,255), 0)
                cv2.putText(image, 'Clap to start a Game',
                            (w//2-300, h//2-85),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (51, 102, 153), 3, cv2.LINE_AA)
                cv2.putText(image, 'Please keep some distance or adjust your webcam',
                            (w//2-210, h//2+130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, 'to locate your face in circle',
                            (w//2-110, h//2+160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                overlay(image, w//2-230, h//2-185, 50, 50, clap_image)
                # print(get_distance(righthand, lefthand))

                if get_distance(righthand, lefthand) < 80 and ( w//2-100 < nose[0] < w//2+100 and h//2-100 < nose[1] < h//2+100):
                    game_start_event = True
                    start_time = time.time()
                    




            if game_start_event == True and time_remaining > 0:
                # print('남은시간', time_remaining)
                # print('\n현재시간',present_time)
                # print('\n시작',start_time)
                time_remaining = int(time_given - (present_time - start_time))

                if (rx0-50 < righthand[0] < rx0+50 and ry0-50 < righthand[1] < ry0+50) 
                    or (rx0-50 < lefthand[0] < rx0+50 and ry0-50 < lefthand[1] < ry0+50) :
                    overlay(image, rx0, ry0, 50, 50, shine_image)
                    score += 1
                    whack_sound.play()
                    rx0=random.randint(50, 590)
                    ry0=random.randint(50, 430)
                    r0.append(rx0)
                    r0.append(ry0)
                 
                if (rx1-50 < righthand[0] < rx1+50 and ry1-50 < righthand[1] < ry1+50) 
                    or (rx1-50 < lefthand[0] < rx1+50 and ry1-50 < lefthand[1] < ry1+50) :
                    overlay(image, rx1, ry1, 50, 50, shine_image)
                    score += 1
                    whack_sound.play()
                    rx1=random.randint(50, 590)
                    ry1=random.randint(50, 430)
                    r1.append(rx0)
                    r1.append(ry0)

                cv2.putText(image, 'Score:',
                            (w//2-250, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 204), 2, cv2.LINE_AA)      

                cv2.putText(image, str(score),
                           (w//2-130, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 204), 2, cv2.LINE_AA)         

                cv2.putText(image, 'Time left:',
                          (w//2+30, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 204), 2, cv2.LINE_AA)      

                cv2.putText(image, str(time_remaining),
                         (w//2+230, 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 204), 2, cv2.LINE_AA) 



                # image, x, y, w, h, overlay_image (좌측 최상단x,y가 50, 50임) 최하단 430 최우측 590
                overlay(image, rx0, ry0, 50, 50, mole_image)
                overlay(image, rx1, ry1, 50, 50, mole_image)                
            
            elif game_start_event == True and time_remaining <= 0:
                
                time_remaining = 0
                game_over_event = True





        except:
            cv2.putText(image, 'Please show your face and keep some distance from your webcam',
            (w//2-260, h//2+220),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            pass    

        # cv2.rectangle(image, (int(righthand[0])-50, int(righthand[1])-50), (int(righthand[0])+50, int(righthand[1])+50), (255,255,255), -1)
        # cv2.rectangle(image, (rx1-50, ry1-50), (rx1+50, ry1+50), rect_color1, -1)          




            # image[ry0:ry0+moleh, rx0:rx0+molew] = mole_image

            # image[ry1:ry1+moleh, rx1:rx1+molew] = mole_image

            # image[10:moleh+10, 20:molew+20] = mole_image


        # 게임종료시에만 실행
        if game_over_event == True:
            
            cv2.rectangle(image, (w//2-170, h//2-130), (w//2+170, h//2+40), (0,0,0), -1)

            cv2.putText(image, 'Game Over',
            (w//2-147, h//2-65),
            cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3, cv2.LINE_AA)
            
            cv2.putText(image, 'Your Score:',
            (w//2-120, h//2),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 

            cv2.putText(image, str(score),
            (w//2+80, h//2+3),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)   


        # Render detections

        # mp_drawing.draw_landmarks(
        #     image,
        #     results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())  



        cv2.imshow('Whack-A-Mole Game with Mediapipe Pose', cv2.resize(image, None, fx=2.0, fy=2.0))  # 화면크기 2배 키움
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
