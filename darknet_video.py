from ctypes import *
import random
import os
import cv2
import time
import darknet
import numpy as np
import argparse
import array
import serial
from threading import Thread, enumerate
from queue import Queue
serial_name = '/dev/usbcam'



#####################################
THRESHOLD=85                        
DEPTH_COEFFICIENT=0.09              
X_BIAS=0                            
X_DEVIATION=15     
#####################################





class RB(object):
    def __init__(self):
        self.classes = []
        self.scores = []
        self.boxes = np.array([])
        self.scores = []
        self.x = 0
        self.y = 0
        self.d = 0
        self.s = 0
        self.img =np.ones((3,3),dtype=np.uint8)
        self.cmd = 'b'
        self.can_be_sent = False
        self.fps = 0
        self.depth_coefficient=0
        self.select_threshold = 0
        self.windowname = 'find_ball'
        self.windowname_= "setting"
        self.x_bias=0
        self.x_deviation=0

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=2,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="./model/yolov4-tiny_last4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4-tiny-ball.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.85,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue,rb):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=rb.select_threshold/100)
        #print(detections,'detections')
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        #print("FPS: {}".format(fps))
        darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue,rb):
    def nothing(x):
        pass
    cv2.namedWindow(rb.windowname,0)
    cv2.namedWindow(rb.windowname_,0)
    cv2.createTrackbar('thresh',rb.windowname_,int(rb.select_threshold),100,nothing)
    cv2.createTrackbar('depth',rb.windowname_,50,100,nothing)
    cv2.createTrackbar('x-bias',rb.windowname_,50,100,nothing)
    cv2.createTrackbar('x-deviation',rb.windowname_,X_DEVIATION,100,nothing)
    def c_dis(cclass, r, f): 
        if cclass == 1:
            d = 2460 * f / r           
        elif cclass == 2:
            d = 2100 * f / r
        return d
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (darknet_width, darknet_height))
    while cap.isOpened():
        rb.depth_coefficient=cv2.getTrackbarPos('depth',rb.windowname_)/50*DEPTH_COEFFICIENT
        rb.select_threshold=cv2.getTrackbarPos('thresh',rb.windowname_)
        rb.x_bias=int(cv2.getTrackbarPos('x-bias',rb.windowname_)-50)+X_BIAS
        rb.x_deviation=cv2.getTrackbarPos('x-deviation',rb.windowname_)
        frame = frame_queue.get()
        frame=cv2.resize(frame,(1280,960))
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        if frame is not None:
            if rb.cmd == '5' or rb.cmd == 'b':
                label_detect="basketball"
                bclass=1
            if rb.cmd == '6' or rb.cmd == 'v':
                label_detect="volleyball" 
                bclass=2
            detections=[n for n in detections if n[0]==label_detect]
            _detections=[]
            for a,b,x in detections:
                _detections.append(x[2]*x[2]+x[3]*x[3])
            try:
                detections=detections[_detections.index(max(_detections))]
                #print(_detections)
            except:
                pass
            #_detections=[(x[0]-x[2])*(x[0]-x[2])+(x[1]-x[3])*(x[1]-x[3]) for x in list(detections)[:,2]]
            #print(_detections,'dddddd')
            if(len(detections)):

                label, confidence, bbox = detections[0],detections[1],detections[2]
                
                bbox_adjusted = convert2original(frame, bbox)
                rb.x=512-int(512*bbox_adjusted[0]/frame.shape[1])+rb.x_bias
                rb.y=int(bbox_adjusted[1])
                rb.d = c_dis(bclass, bbox_adjusted[2]/frame.shape[1],rb.depth_coefficient)
                    
                #print(bbox_adjusted,'origin')
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
                s="s:"+confidence
            else:
                s="s:NONE" 
                rb.x=0
                rb.y=0
                rb.d=0
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            if rb.cmd == '5':
                s1 = 'cmd:5 finding basketball'
            elif rb.cmd == '6':
                s1 = 'cmd:6 finding volleyball'
            elif rb.cmd == 'b':
                s1 = '1-3 basketball'
            elif rb.cmd == 'v':
                s1 = '4-6 volleyball'
            else:
                s1 = 'wrong cmd %s' %rb.cmd
            cv2.putText(image, s1,(20,50), cv2.FONT_HERSHEY_DUPLEX,2, (28,150,36), 2)
            cv2.putText(image,s,(500,150), cv2.FONT_HERSHEY_DUPLEX,2, (18,87,220), 2)
            cv2.putText(image,"x:%d"%rb.x,(20,150), cv2.FONT_HERSHEY_DUPLEX,2, (18,87,220), 2)
            cv2.putText(image,"d:%d"%rb.d,(220,150), cv2.FONT_HERSHEY_DUPLEX,2, (18,87,220), 2)
            cv2.putText(image, "fps:%.1f"%fps,(20,250), cv2.FONT_HERSHEY_DUPLEX,2, (254,67,101), 2)
            cv2.putText(image, "thresh:%d f:%.2f bias:%d"%(rb.select_threshold,rb.depth_coefficient,rb.x_bias),(20,350), cv2.FONT_HERSHEY_DUPLEX,2, (100,100,100), 2)
            cv2.resizeWindow(rb.windowname,800, 600)    # 设置长和宽
            y_min=400
            cv2.line(image,(0,y_min),(image.shape[1],y_min),(0,0,255),2,8)
            cv2.line(image,(int(image.shape[1]/2)+rb.x_bias,0),(int(image.shape[1]/2+rb.x_bias),image.shape[0]),(0,0,255),2,8)
            if not args.dont_show:
                cv2.imshow(rb.windowname, image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if cv2.waitKey(fps) & 0xff == ord('q'):
                break
            if cv2.waitKey(fps) & 0xff == ord('v'):
                rb.cmd = 'v'
            if cv2.waitKey(fps) & 0xff == ord('b'):
                rb.cmd = 'b'
    cap.release()
    video.release()
    cv2.destroyAllWindows()

def serial_(rb):
    
    buf = array.array('B', [0] * 10)
    # 10个16进制数组 buf[0]-buf[4] 固定
    # buf[5] buf[6] x
    # buf[7]buf[8] d
    buf[0] = 0x40
    buf[1] = 0x5E
    buf[2] = 0x76
    buf[3] = 0x00
    buf[4] = 0x00



    ser = serial.Serial(serial_name,115200, timeout=0.5)

    def send_data(x, d):
        buf[5] = int(hex(x // 256),16)
        buf[6] = int(hex(x % 256),16)
        buf[7] = int(hex(d // 256),16)
        buf[8] = int(hex(d % 256),16)
        suma = 0
        for i in range(9):
            suma = suma + buf[i]
        buf[9] = (suma) & 0xff
        ser.write(buf)
    print("正在发送数据... 波特率：%d"%ser.baudrate)
    while 1:
        try:
            if ser.inWaiting() > 0:
                rb.cmd = ser.read(1).decode()
            time.sleep(0.1)
            if (rb.y > 400) & (rb.d < 4000):
                rb.can_be_sent = True
            else:
                rb.can_be_sent = False
            rb.x = change_rbx(rb,X_DEVIATION)
            if rb.can_be_sent:
                send_data(rb.x,int(rb.d))

            else:
                send_data(0,0)
            #print((rb.x,rb.y,rb.d,rb.can_be_sent))
        except serial.SerialException as e:
                # There is no new data from serial port
                print("串口断开..")
                time.sleep(5)
                for i in range(10):
                     ser = serial.Serial(serial_name,115200, timeout=1)


        except TypeError as e:
		# Disconnect of USB->UART occured
            return None
        else:
            continue
    
def check_ball_pos(rb,y_min=400,d_min = 2000):
    if (rb.y > y_min) & (rb.d < d_min):
        rb.can_be_sent = True
    else:
        rb.can_be_sent = False

def change_rbx(rb,x):
    if abs(rb.x-256) < x:
        return 256
    else:
        return rb.x

if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    rb = RB()
    rb.select_threshold=THRESHOLD
    rb.depth_coefficient=DEPTH_COEFFICIENT
    rb.x_bias=X_BIAS
    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    
    cap = cv2.VideoCapture(1)
    cap.set(3,640)
    cap.set(4,480)
    Thread(target=serial_,args=(rb,)).start()
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue,rb)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue,rb)).start()
