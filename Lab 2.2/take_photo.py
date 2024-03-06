import cv2
import uuid

cap = cv2.VideoCapture(0, cv2.CAP_FIREWIRE)
ret, frame = cap.read()
cv2.imshow("img", frame)
cv2.imwrite(f'{uuid.uuid4()}.png', frame)
cap.release()