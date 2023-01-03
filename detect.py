"""
origin: https://github.com/ShiqiYu/libfacedetection/tree/master/opencv_dnn
"""
import argparse
import numpy as np
import cv2 as cv
import os
import glob
import tqdm


def str2bool(v: str) -> bool:
    if v.lower() in ['true', 'yes', 'on', 'y', 't']:
        return True
    elif v.lower() in ['false', 'no', 'off', 'n', 'f']:
        return False
    else:
        raise NotImplementedError


## face[0]~face[3]: [tof-left, width, height]
## face[4]~face[13]: 5 landmark points 
## face[-1]: confident score
def visualize(image, faces, print_flag=False, fps=None):
    output = image.copy()

    if fps:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    for idx, face in enumerate(faces):
        if print_flag:
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

        coords = face[:-1].astype(np.int32)
        # Draw face bounding box
        cv.rectangle(output, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
        # Draw landmarks
        cv.circle(output, (coords[4], coords[5]), 2, (255, 0, 0), 2)
        cv.circle(output, (coords[6], coords[7]), 2, (0, 0, 255), 2)
        cv.circle(output, (coords[8], coords[9]), 2, (0, 255, 0), 2)
        cv.circle(output, (coords[10], coords[11]), 2, (255, 0, 255), 2)
        cv.circle(output, (coords[12], coords[13]), 2, (0, 255, 255), 2)
        # Put score
        cv.putText(output, '{:.4f}'.format(face[-1]), (coords[0], coords[1]+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    return output


## face[0]~face[3]: [tof-left, width, height]
## face[4]~face[13]: 5 landmark points 
## face[-1]: confident score
def crop(image, faces, print_flag=False, fps=None):
    h, w, _ = image.shape
    savefaces = []
    if faces is None:
        return savefaces

    for idx, face in enumerate(faces):
        face = face.astype(np.int32)
        ## assert index is right
        if face[0] >=0 and face[1] >= 0 and face[2] >=0 and face[3] >= 0 and face[1]+face[3] < h and face[0]+face[2] < w:
            temp = image[face[1]:face[1]+face[3], face[0]:face[0]+face[2], :]
            savefaces.append(temp)
    return savefaces


## for IEMOCAP
def crop_left_right(image, faces, print_flag=False, fps=None):
    h, w, _ = image.shape
    leftfaces = []
    rightfaces = []
    if faces is None:
        return leftfaces, rightfaces

    for idx, face in enumerate(faces):
        face = face.astype(np.int32)
        ## assert index is right
        if face[0] >=0 and face[1] >= 0 and face[2] >=0 and face[3] >= 0 and face[1]+face[3] < h and face[0]+face[2] < w:
            temp = image[face[1]:face[1]+face[3], face[0]:face[0]+face[2], :]
            if face[0]+face[2] < (w/2):
                leftfaces.append(temp)
            else:
                rightfaces.append(temp)
    return leftfaces, rightfaces


def main():
    backends = (cv.dnn.DNN_BACKEND_DEFAULT,
                cv.dnn.DNN_BACKEND_HALIDE,
                cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
                cv.dnn.DNN_BACKEND_OPENCV)
    targets = (cv.dnn.DNN_TARGET_CPU,
               cv.dnn.DNN_TARGET_OPENCL,
               cv.dnn.DNN_TARGET_OPENCL_FP16,
               cv.dnn.DNN_TARGET_MYRIAD)

    parser = argparse.ArgumentParser(description='A demo for running libfacedetection using OpenCV\'s DNN module.')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help='Choose one of computation backends: '
                             '%d: automatically (by default), '
                             '%d: Halide language (http://halide-lang.org/), '
                             '%d: Intel\'s Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), '
                             '%d: OpenCV implementation' % backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Choose one of target computation devices: '
                             '%d: CPU target (by default), '
                             '%d: OpenCL, '
                             '%d: OpenCL fp16 (half-float precision), '
                             '%d: VPU' % targets)
    # Location
    parser.add_argument('--video', type=str, help='Path to the video')
    parser.add_argument('--videofolder', type=str, help='Path to the video')
    parser.add_argument('--model', type=str, help='Path to .onnx model file.')
    parser.add_argument('--dataset', type=str, help='Which dataset is processed.')
    # Inference parameters
    parser.add_argument('--score_threshold', default=0.95, type=float, help='Threshold for filtering out faces with conf < conf_thresh.')
    parser.add_argument('--nms_threshold', default=0.3, type=float, help='Threshold for non-max suppression.')
    parser.add_argument('--top_k', default=700, type=int, help='Keep keep_top_k for results outputing.')
    # Result
    parser.add_argument('--save', type=str, help='Path to save faces.')
    args = parser.parse_args()

    # Instantiate yunet
    yunet = cv.FaceDetectorYN.create(
        model=args.model,
        config='',
        input_size=(320, 320),
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        top_k=args.top_k,
        backend_id=args.backend,
        target_id=args.target
    )

    if args.videofolder is None and args.video is not None:
        videoList = [args.video]
    elif args.videofolder is not None and args.video is None:
        videoList = glob.glob(args.videofolder+'/*.mp4')
    else:
        print ('input is not satisfied requirement.')

    for videopath in tqdm.tqdm(videoList):
        videoname = os.path.basename(videopath)[:-4]

        ## define folders
        if not os.path.exists(args.save): os.makedirs(args.save)
        videosave = os.path.join(args.save, videoname)
        if not os.path.exists(videosave): os.makedirs(videosave)

        ## process for args.video
        print('%s begin.' %(videopath))
        cap = cv.VideoCapture(videopath)
        frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        yunet.setInputSize([frame_w, frame_h])

        index = 1
        while 1:
            ## read frame and detect face
            has_frame, frame = cap.read()
            if not has_frame:
                print('All frames grabbed!')
                break
            _, faces = yunet.detect(frame) # # faces: None, or nx15 np.array

            ########################################
            if args.dataset == 'IEMOCAP': ## save left and right faces
                left_faces, right_faces = crop_left_right(frame, faces)
                left_gender = videoname[5] # 'M' or 'F'
                if left_gender == 'M': right_gender = 'F'
                if left_gender == 'F': right_gender = 'M'

                framesave = os.path.join(videosave, '%06d' %(index))
                leftfacesave = '%s_%s.jpg' %(framesave, left_gender)
                rightfacesave = '%s_%s.jpg' %(framesave, right_gender)
                if len(left_faces) == 0:
                    face = np.zeros((100, 100, 3))
                    cv.imwrite(leftfacesave, face)
                else:
                    face = left_faces[0]
                    cv.imwrite(leftfacesave, face)

                if len(right_faces) == 0:
                    face = np.zeros((100, 100, 3))
                    cv.imwrite(rightfacesave, face)
                else:
                    face = right_faces[0]
                    cv.imwrite(rightfacesave, face)
            ########################################
                
            ## frame index add 1
            index += 1
        print('%s finished.' %(videopath))
        
        
if __name__ == '__main__':
    main()