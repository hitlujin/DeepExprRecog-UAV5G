import insightface
from skimage import transform
import cv2
import numpy as np
det = insightface.model_zoo.SCRFD("det.onnx")


expression_name = ["happy","disgust"]
det = insightface.model_zoo.SCRFD("det.onnx")


def face_align_landmarks_sk(img, landmarks, image_size=(112, 112), method="similar"):
    tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()
    src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32)
    ret = []
    for landmark in landmarks:
        # landmark = np.array(landmark).reshape(2, 5)[::-1].T
        tform.estimate(landmark, src)
        ret.append(transform.warp(img, tform.inverse, output_shape=image_size))
    ret = np.transpose(ret, axes=[0,3,1,2])
    return (np.array(ret) * 255).astype(np.uint)

def do_detect_in_image(image, det, image_format="BGR"):
    imm_BGR = image if image_format == "BGR" else image[:, :, ::-1]
    imm_RGB = image[:, :, ::-1] if image_format == "BGR" else image
    bboxes, pps = det.detect(imm_BGR, input_size = (640,640))
    nimgs = face_align_landmarks_sk(imm_RGB, pps)
    bbs, ccs = bboxes[:, :4].astype("int"), bboxes[:, -1]
    return bbs, ccs,pps, nimgs


# read image
img = cv2.imread("mul1.jpg")
print(img.shape)

def print_image(frame,saveImagePath):
    #detect face and save path
    bbs,ccs,pps,imgs = do_detect_in_image(img,det)
    #draw box on image with nice color
    for i in range(len(bbs)):
        cv2.rectangle(frame, (bbs[i][0], bbs[i][1]), (bbs[i][2], bbs[i][3]), (0, 255, 0), 2)
        cv2.putText(frame, expression_name[int(ccs[i])], (bbs[i][0], bbs[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(saveImagePath, frame)
print_image(img,"test.jpg")


