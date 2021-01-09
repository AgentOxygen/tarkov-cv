# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:32:11 2020

@author: Cameron Cummins 
"""
import streamlink as sl
import cv2
import numpy as np
from PIL import Image
import os
import time as t
import queue
from threading import Thread
import Analysis as al

# Returns video clip from URL head that contains the specified time
# Example:
#    url_head = 'https://dqrpb9wgowsf5.cloudfront.net/bc2b2d73bb19dd89ee64_lvndmark_40002390973_1609189430/480p30'
#    time_s = 23
#  Returns:
#    File at 'https://dqrpb9wgowsf5.cloudfront.net/bc2b2d73bb19dd89ee64_lvndmark_40002390973_1609189430/480p30/230.ts' or 'http.../480p30/230-muted.ts'
#    or returns None if failed
def getTwitchClip(url_head, time_s):
    clip_num = int((time_s - (time_s % 10)) / 10)
    ret_video = cv2.VideoCapture(url_head + "/" + str(clip_num) + ".ts")
    if not ret_video.isOpened(): 
        ret_video = cv2.VideoCapture(url_head + "/" + str(clip_num) + "-muted.ts")
    return ret_video

# Returns resolution size that corresponds to the specified bitrate
def getResFromBitrate(bitrate):
    if bitrate == '720p':
        return (1280, 720)
    elif bitrate == '480p':
        return (852, 480)
    elif bitrate == '360p':
        return (640, 360)
    elif bitrate == '240p':
        return (426, 240)
    else:
        return (0, 0)

# Returns URL parsed up one directory
# Example:
#      source_master = "http://wwww.google.com/dir1/webpage.html"
#   Returns:
#      "http://wwww.google.com/dir1"
def getDirectory(source_master):
    for index in range(len(source_master) - 1, 0, -1):
        if source_master[index] == '/':
            return source_master[0:index]

def getClockFromSeconds(time_s):
    h = int(np.floor(time_s / (60*60)))
    m = int(np.floor(time_s / (60))) - h*60
    s = int(np.floor(time_s)) - h*60*60 - m*60
    return str(h) + "h" + str(m) + "m" + str(s) + "s"

# conductAnalysis(getFrames("https://www.twitch.tv/videos/863546471", 56*60, 3*60))
# Should find
# Safe -> SSD and Dollars
# Safe -> Diary and SSD
# Safe -> Two rouble stacks
# PC Block -> Cord

template_dir = "templates/"
output_dir_menu = "output/"

menu_template = cv2.imread(template_dir + "main_menu_template.png", cv2.IMREAD_GRAYSCALE)
main_menu_w, main_menu_h = menu_template.shape[::-1]
menu_stash_template = cv2.imread(template_dir + "menu_stash_template.png", cv2.IMREAD_GRAYSCALE)
menu_stash_w, menu_stash_h = menu_stash_template.shape[::-1]
menu_loading_stash_template = cv2.imread(template_dir + "menu_loading_stash_template.png", cv2.IMREAD_GRAYSCALE)
menu_loading_stash_w, menu_loading_stash_h = menu_loading_stash_template.shape[::-1]
loot_template = cv2.imread(template_dir + "loot_template.jpg", cv2.IMREAD_GRAYSCALE)
loot_w, loot_h = loot_template.shape[::-1]
squareloot_template = cv2.imread(template_dir + "squareloot_template.png", cv2.IMREAD_GRAYSCALE)
squareloot_w, squareloot_h = squareloot_template.shape[::-1]
search_loot_template = cv2.imread(template_dir + "search_loot_template.jpg", cv2.IMREAD_GRAYSCALE)
search_loot_w, search_loot_h = search_loot_template.shape[::-1]
loading_template = cv2.imread(template_dir + "raid_loading_template.jpg", cv2.IMREAD_GRAYSCALE)
loading_w, loading_h = loading_template.shape[::-1]
searching_template = cv2.imread(template_dir + "searching_template.jpg", cv2.IMREAD_GRAYSCALE)
searching_w, searching_h = searching_template.shape[::-1]
unknown_template = cv2.imread(template_dir + "unknown_item_template.jpg", cv2.IMREAD_GRAYSCALE)
unknown_w, unknown_h = unknown_template.shape[::-1]
unsearched_template = cv2.imread(template_dir + "unsearched_template.jpg", cv2.IMREAD_GRAYSCALE)
unsearched_w, unsearched_h = unsearched_template.shape[::-1]

menu_load_match_threshold = 0.75
loot_match_threshold = 0.85
frame_rate = 30
bitrate = '720p'

multithreading_blocking_timeout = 10
multithreading_download_workers = 1
multithreading_decoding_workers = 5

# Debugging function useful for testing stuff
def debugTemplateMatching(source_head, time_sec, duration_sec, template, threshold):
    w, h = template.shape[::-1]
    ret_frames = []
    skips = 2
    for index in range(int(duration_sec / 10)):
        print(str(index) + " out of " + str(int(duration_sec / 10)))
        index += int(time_sec / 10)
        segment = cv2.VideoCapture(source_head + "/" + str(index) + ".ts")
        if not segment.isOpened(): 
            segment = cv2.VideoCapture(source_head + "/" + str(index) + "-muted.ts")
        empty_frames = 0
        while True:
            for x in range(skips):
                segment.grab()
            ret, frame = segment.read()
            if frame is not None:
                res = cv2.matchTemplate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val >= threshold:
                    top_left = max_loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    cv2.rectangle(frame,top_left, bottom_right, (0, 0, 255), 2)
                ret_frames.append(frame)
                
                empty_frames = 0
            else:
                empty_frames += 1
                if empty_frames > 10:
                    break
        display = True
    while display and len(ret_frames) > 0:
        for frame in ret_frames:
            cv2.imshow('Debug Frames', frame)
            if cv2.waitKey(int((1000/frame_rate) * skips)) & 0xFF == ord('s'): 
                display = False
                break
    cv2.destroyAllWindows()

# Conducts a template match for the specified threshold, optionally outputting the results in an image    
def matchAndOutput(crop, template, template_w, template_h, name, threshold, output=True):
    res = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= threshold:
        # Find file name and output image
        if output:
            index = 0
            while os.path.isfile('output/' + name + '-' + str(index) + '.jpg'):
                index += 1
            cv2.imwrite('output/' + name + '-' + str(index) + '.jpg', crop)
            print("Match for '" + name + " found!")
        return True
    return False

class DecodingWorker(Thread):
    def __init__(self, queue_read, queue_write):
        Thread.__init__(self)
        self.queue_read = queue_read
        self.queue_write = queue_write
    def run(self):
        empty_frame_limit = 20
        while True:
            # Get segment and read first frame
            try:
                time_i, captured_segment = self.queue_read.get(True, multithreading_blocking_timeout)
            except queue.Empty:
                #print("Decoding worker quitting...")
                break
            ret, frame = captured_segment.read()
            # Keep reading until a non-empty frame is found in case of errors
            empty_frames = 0
            error_flag = False
            while frame is None:
                if empty_frames < empty_frame_limit:
                    ret, frame = captured_segment.read()
                    empty_frames += 1
                else:
                    error_flag = True
                    break
            if error_flag: 
                print("(" + str(time_i) + ") Empty frame limit exceeded! Segment at time = " + getClockFromSeconds(time_i*10) + " dumped.")
                captured_segment.release()
                self.queue_read.task_done()
                continue
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            xmax, ymax = gray_frame.shape[::-1]
            
            # Check for main menu or loading screen, scrap segment if true
            menu_crop = gray_frame[680:ymax, 580:xmax]
            loading_crop = gray_frame[0:120, 420:850]
            stash_crop = gray_frame
            if matchAndOutput(menu_crop, menu_template, main_menu_w, main_menu_h, "main_menu", menu_load_match_threshold, False):
                print("(" + str(time_i) + ") Menu detected in segement at time = " + getClockFromSeconds(time_i*10) + ", skipping")
                captured_segment.release()
                self.queue_read.task_done()
                continue
            elif matchAndOutput(loading_crop, loading_template, loading_w, loading_h, "loading", menu_load_match_threshold, False):
                print("(" + str(time_i) + ") Loading detected in segement at time = " + getClockFromSeconds(time_i*10) + ", skipping")
                captured_segment.release()
                self.queue_read.task_done()
                continue
            
            # If this is a good segment, start filtering this segment for good frames (those with "Loot" tag displayed)
            good_frames = []
            empty_frames = 0
            max_frames = frame_rate * 10
            n_frames = 0
            gap = 0
            while True:
                ret, frame = captured_segment.read()
                n_frames += 1
                if frame is not None:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    searching_crop = gray_frame[30:90, 790:850]
                    menu_crop = gray_frame[680:ymax, 580:xmax]
                    
                    if matchAndOutput(searching_crop, squareloot_template, squareloot_w, squareloot_h, "squareloot", loot_match_threshold, False):
                        if not matchAndOutput(menu_crop, menu_template, main_menu_w, main_menu_h, "main_menu", menu_load_match_threshold, False):
                            crop_frame = gray_frame[30:425, 840:1270]
                            good_frames.append(crop_frame)
                    else:
                        gap += 1
                        if len(good_frames) > 2 and gap > 10:
                            self.queue_write.put((time_i, good_frames))
                            time_i += 0.01
                            good_frames = []
                            gap = 0
                    empty_frames = 0
                else:
                    empty_frames += 1
                    if empty_frames > empty_frame_limit:
                        break
                if n_frames > max_frames:
                    break
            print("(" + str(time_i) + ") Finished decoding segement at time = " + getClockFromSeconds(time_i*10))
            captured_segment.release()
            if len(good_frames) > 0:
                self.queue_write.put((time_i, good_frames))
            self.queue_read.task_done()

class DownloadWorker(Thread):
    def __init__(self, queue_read, queue_write):
        Thread.__init__(self)
        self.queue_read = queue_read
        self.queue_write = queue_write
    def run(self):
        while True:
            try:
                index, url_head = self.queue_read.get(True, multithreading_blocking_timeout)
            except queue.Empty:
                #print("Download worker quitting...")
                break
            segment = cv2.VideoCapture(url_head + "/" + str(index) + ".ts")
            if not segment.isOpened(): 
                segment = cv2.VideoCapture(url_head + "/" + str(index) + "-muted.ts")
            print("Downloaded index " + str(index))
            self.queue_write.put((index, segment))
            self.queue_read.task_done()
            
def getFrames(streamURL, time_s, duration_s):

    source_master = sl.streams(streamURL)[bitrate].url
    source_head = getDirectory(source_master)
    
    print("Start!")
    ts = t.time()
    # Create queues
    url_queue = queue.Queue()
    capture_queue = queue.Queue()
    frame_queue = queue.Queue()

    # Create Download Workers
    for x in range(multithreading_download_workers):
        worker = DownloadWorker(url_queue, capture_queue)
        worker.daemon = True
        worker.start()

    # Create Decoding Workers
    for x in range(multithreading_decoding_workers):
        worker = DecodingWorker(capture_queue, frame_queue)
        worker.daemon = True
        worker.start()

    # Fill URL queue
    print("Initializing URL queue...")
    for index in range(int(duration_s / 10)):
        url_queue.put((index + int(time_s / 10), source_head))

    # Wait for downloading to finish
    url_queue.join()
    print("URL queue finished.")    
    
    # Wait for decoding to finish
    capture_queue.join()
    print("Capture queue finished.")

    final_frames = []
    print("Getting final frames...")
    
    while not frame_queue.empty():
        queue_grabbed = frame_queue.get()
        flag = True
        for index in range(len(final_frames)):
            if queue_grabbed[0] < final_frames[index][0]:
                final_frames.insert(index, queue_grabbed)
                flag = False
                break
        if flag:
            final_frames.append(queue_grabbed)
        frame_queue.task_done()
    frame_queue.join()

    time_to_execute = round(t.time() - ts, 3)
    time_of_video = duration_s
    
    print("============= Downloading and Decoding Report =============")
    print("URL: " + streamURL)
    print("time_s: " + str(time_s) + "  duration_s:" + str(duration_s))
    print("All tasks complete: " + str(time_of_video) + " seconds decoded. ")
    print("Execution time = " + str(time_to_execute) + "s")
    print("Video Time to Processing Time Ratio (s): " + str(round(time_of_video / time_to_execute, 2)))
    
    return final_frames