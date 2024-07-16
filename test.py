import cv2
import face_recognition
import numpy as np
import time
from picamera2 import Picamera2
from libcamera import controls
from pymycobot.mycobot import MyCobot

# Picamera2 초기화
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
picam2.configure(config)
picam2.start()
mycobot=MyCobot("/dev/ttyACM0")

# 자동 화이트밸런스 및 자동 노출 설정
picam2.set_controls({"AwbEnable": True, "AeEnable": True})

# 샘플 사진을 로드하고 인식 방법을 학습
known_image = face_recognition.load_image_file("wooheum.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# 알려진 얼굴 인코딩과 그들의 이름으로 배열 생성
known_face_encodings = [known_face_encoding]
known_face_names = ["wooheum"]

# 프레임 처리 간격 설정 (초)
frame_interval = 0.1
previous_time = time.time()

while True:
    # 현재 시간 확인
    current_time = time.time()

    # 설정된 간격마다 프레임 처리
    if current_time - previous_time >= frame_interval:
        previous_time = current_time

        # 이미지 캡처
        image = picam2.capture_array()

        # RGB로 변환 (face_recognition은 RGB 형식을 사용)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # 프레임 크기를 절반으로 줄임 (추가적인 성능 향상을 위해)
        small_frame = cv2.resize(rgb_image, (0, 0), fx=0.5, fy=0.5)

        # 현재 프레임에서 모든 얼굴 위치 찾기
        face_locations = face_recognition.face_locations(small_frame, model="hog")

        if face_locations:
            # 가장 큰 얼굴 선택 (가장 가까운 얼굴)
            face_location = max(face_locations, key=lambda rect: (rect[2] - rect[0]) * (rect[1] - rect[3]))
            
            # 선택된 얼굴의 인코딩 찾기
            face_encoding = face_recognition.face_encodings(small_frame, [face_location], model="small")[0]

            # 크기를 조정했으므로 얼굴 위치를 2배로 조정
            top, right, bottom, left = [coord * 2 for coord in face_location]

            # 얼굴이 알려진 얼굴과 일치하는지 확인
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            similarity = (1 - face_distances[best_match_index]) * 100

            if similarity > 60:  # 60% 이상의 유사도를 가질 때 일치로 간주
                name = known_face_names[best_match_index]
                mycobot.send_angle([0, 0, 30, 0, 0, 0], 50)
                time.sleep(5)
                mycobot.send_angle([0, 0, 0, 0, 0, 0], 50)
            else:
                name = "Unknown"

            # 얼굴 주위에 사각형을 그리고 라벨 지정
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # 이름과 유사도를 함께 표시
            label = f"{name} ({similarity:.2f}%)"
            cv2.putText(image, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 결과 프레임 표시
        cv2.imshow('Video', image)

    # 키보드에서 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 창 닫기
cv2.destroyAllWindows()
picam2.stop()
