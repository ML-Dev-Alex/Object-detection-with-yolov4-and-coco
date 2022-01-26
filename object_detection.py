import cv2
from gui_buttons import Buttons

# Initialize Buttons
button = Buttons()
button.add_button("Person", 20, 20)
button.add_button("Cell Phone", 20, 100)
button.add_button("Umbrella", 20, 180)
button.add_button("Book", 20, 260)
button.add_button("Remote", 20, 340)
colors = button.colors
classes = []


with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


# Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights",
                      "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(512, 512), scale=1/255)

# Initialize camera and set size
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

while True:
    ret, frame = cap.read()

    active_buttons = button.active_buttons_list()

    # Object Detection
    (class_ids, scores, bboxes) = model.detect(
        frame, confThreshold=0.3, nmsThreshold=.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        color = colors[class_id]
        # color = (255, 52, 0)
        if class_name in active_buttons:
            cv2.putText(frame, str(classes[class_id]), (x, y-10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 50), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

    button.display_buttons(frame)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
