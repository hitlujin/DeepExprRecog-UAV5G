import insightface
import cv2
import numpy as np



class DetFace():
    def __init__(self):
        self.det = insightface.model_zoo.SCRFD("det.onnx")
        self.det.prepare(-1)

    def detect_matimg_result(self,mat_img):
        bboxes, pps = self.det.detect(mat_img, (640, 640))
        face = []
        for bbox,pp in zip(bboxes,pps):
            face_feature = {}
            face_feature["bbox"] = [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])]
            face_feature["face_detect_score"] = float(bbox[4])
            landmark = []
            for i in range(5):
                landmark.append([int(pp[i][0]),int(pp[i][1])])
            face_feature["landmark"] = landmark
            face.append(face_feature)
        return face

    def test(self,file_path):
        img = cv2.imread(file_path)
        bboxes, pps = self.det.detect(img, (640, 640))
        for bbox in bboxes:
            cv2.rectangle(img,(int(bbox[0]),int(bbox[1]),int(bbox[2] - bbox[0]),int(bbox[3] - bbox[1])),(0,255,0),1,4)
        for landmarks in pps:
            for dot in landmarks:
                cv2.circle(img,(int(dot[0]),int(dot[1])),1,(0,255,0),1)
        cv2.imshow('test',img)
        cv2.waitKey()

# detface = DetFace()
# detface.test('mul.jpg')
# img = cv2.imread('mul.jpg')
# print(detface.detect_matimg_result(img))