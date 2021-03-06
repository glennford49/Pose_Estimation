import logging
import os.path as osp
import sys
from itertools import cycle
from enum import Enum
from time import perf_counter
from human_pose_estimation_demo.model import HPEAssociativeEmbedding, HPEOpenPose
from human_pose_estimation_demo.visualization import show_poses
from helpers import put_highlighted_text
import cv2,time
import numpy as np
from openvino.inference_engine import IECore
import monitors
sys.path.append(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'common'))
#camera = "0" # using webcam
camera = "video/despacito.mp4"
model ="model/human-pose-estimation-0001.xml"
#architecture_type=['ae', 'openpose']
architecture_type= 'openpose'
prob_threshold=0.1
device = "CPU"
num_infer_requests=1
num_streams=''
num_threads= None
loop = 0
utilization_monitors=''
raw_output_message=False
no_show=False
tsize=150
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

class Modes(Enum):
    USER_SPECIFIED = 0
    MIN_LATENCY = 1

class ModeInfo:
    def __init__(self):
        self.last_start_time = perf_counter()
        self.last_end_time = None
        self.frames_count = 0
        self.latency_sum = 0

def get_plugin_configs(device, num_streams, num_threads):
    config_user_specified = {}
    config_min_latency = {}
    devices_nstreams = {}
    if num_streams:
        devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
                           if num_streams.isdigit() \
                           else dict(device.split(':', 1) for device in num_streams.split(','))

    if 'CPU' in device:
        if num_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                                                              if int(devices_nstreams['CPU']) > 0 \
                                                              else 'CPU_THROUGHPUT_AUTO'
        config_min_latency['CPU_THROUGHPUT_STREAMS'] = '1'

    if 'GPU' in device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                                                              if int(devices_nstreams['GPU']) > 0 \
                                                              else 'GPU_THROUGHPUT_AUTO'
        config_min_latency['GPU_THROUGHPUT_STREAMS'] = '1'

    return config_user_specified, config_min_latency

def main():
    
    log.info('Initializing Inference Engine...')
    ie = IECore()
    config_user_specified, config_min_latency = get_plugin_configs(device, num_streams, num_threads)
    log.info('Loading network...')
    completed_request_results = {}
    modes = cycle(Modes)
    prev_mode = mode = next(modes)
    log.info('Using {} mode'.format(mode.name))
    mode_info = {mode: ModeInfo()}
    exceptions = []
    if architecture_type == 'ae':
        HPE = HPEAssociativeEmbedding
    else:
        HPE = HPEOpenPose
    hpes = {
        Modes.USER_SPECIFIED:
            HPE(ie, model, target_size=tsize, device=device, plugin_config=config_user_specified,
                results=completed_request_results, max_num_requests=num_infer_requests,
                caught_exceptions=exceptions),
        Modes.MIN_LATENCY:
            HPE(ie, model, target_size=tsize, device=device.split(':')[-1].split(',')[0],
                plugin_config=config_min_latency, results=completed_request_results, max_num_requests=1,
                caught_exceptions=exceptions)
    }
    try:
        input_stream = int(camera)
    except ValueError:
        input_stream = camera
    cap = cv2.VideoCapture(input_stream)
    wait_key_time = 1
    next_frame_id = 0
    next_frame_id_to_show = 0
    input_repeats = 0
    log.info('Starting inference...')
    presenter = monitors.Presenter(utilization_monitors, 55,
        (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 8)))
    
    while (cap.isOpened() \
        or completed_request_results \
        or len(hpes[mode].empty_requests) < len(hpes[mode].requests)) \
        and not exceptions:
        
        if next_frame_id_to_show in completed_request_results:
            frame_meta, raw_outputs = completed_request_results.pop(next_frame_id_to_show)
            poses, scores = hpes[mode].postprocess(raw_outputs, frame_meta)
            valid_poses = scores > prob_threshold
            poses = poses[valid_poses]
            scores = scores[valid_poses]
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']
            if len(poses) and raw_output_message:
                log.info('Poses:')
            presenter.drawGraphs(frame)
            show_poses(frame, poses, scores, pose_score_threshold=prob_threshold,
                point_score_threshold=prob_threshold)
            if raw_output_message:
                for pose, pose_score in zip(poses, scores):
                    pose_str = ' '.join('({:.2f}, {:.2f}, {:.2f})'.format(p[0], p[1], p[2]) for p in pose)
                    log.info('{} | {:.2f}'.format(pose_str, pose_score))
            next_frame_id_to_show += 1
            if prev_mode == mode:
                mode_info[mode].frames_count += 1
            elif len(completed_request_results) == 0:
                mode_info[prev_mode].last_end_time = perf_counter()
                prev_mode = mode
            
            # Frames count is always zero if mode has just been switched (i.e. prev_mode != mode).
            if mode_info[mode].frames_count != 0:
                fps =mode_info[mode].frames_count / (perf_counter() - mode_info[mode].last_start_time)
                    
                fps_message = 'FPS: {:.1f}'.format(fps)#
                cv2.rectangle(frame,(15,0),(135,24),(0,255,255),-1)
                # Draw performance stats over frame.
                put_highlighted_text(frame, fps_message, (15, 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0), 2)
            if not no_show:
                cv2.imshow('Frame', frame)
                key = cv2.waitKey(wait_key_time)
                ESC_KEY = 27
                
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    
                    break
        elif hpes[mode].empty_requests and cap:
            start_time = perf_counter()
            ret, frame = cap.read()
            if not ret:
                if input_repeats < loop or loop < 0:
                    cap.open(input_stream)
                    input_repeats += 1
                else:
                    cap.release()
                continue
            hpes[mode](frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1
        else:
            hpes[mode].await_any()
        
    if exceptions:
        raise exceptions[0]
    for exec_net in hpes.values():
        exec_net.await_all()
    print(presenter.reportMeans())
    
    cv2.destroyAllWindows()
if __name__ == '__main__':
    sys.exit(main() or 0)
