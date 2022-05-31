import cv2
import mediapipe as mp
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron
# ----------------------------------------------------------------------------------------------------------------------
def ex_static(filename_in):

    objectron = mp_objectron.Objectron(static_image_mode=True,max_num_objects=5,min_detection_confidence=0.5,model_name='Shoe')
    image = cv2.imread(filename_in)
    results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #if not results.detected_objects:
    #    continue
    annotated_image = image.copy()
    for detected_object in results.detected_objects:
        mp_drawing.draw_landmarks(annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
        mp_drawing.draw_axis(annotated_image, detected_object.rotation,detected_object.translation)

    cv2.imwrite(folder_out+'res.png', annotated_image)
    return

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    ex_static('./images/ex_pose/image5.jpg')