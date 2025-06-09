import cv2

# Spróbuj kamery 1, 2, itd. aż znajdziesz Camo
cap = cv2.VideoCapture(1)  # lub 2, 3, 4…

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camo kamera", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()