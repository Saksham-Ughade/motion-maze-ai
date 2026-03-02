import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open nahi ho raha. Try index 1.")
    cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("Camera Test (press q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
