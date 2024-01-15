import cv2
from ultralytics import YOLO

img1 = cv2.imread("test1.jpg")

model = YOLO("last.pt")

camera = cv2.VideoCapture(0)
while camera.isOpened():
    success, frame = camera.read()
    if not success: break

    frame = img1.copy()

    results = model(frame)
    print(len(results))
    for r in results:
        print(r.boxes.xyxy)
        for xyxy in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    key_code = cv2.waitKey(1)
    if key_code in [27, ord('q')]:
        break
camera.release()
cv2.destroyAllWindows()
