# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 12:24:08 2021

@author: Cameron Cummins
"""

import cv2
import numpy as np
from PIL import Image
import random
from os import listdir
from os.path import isfile, join
from skimage.metrics import structural_similarity
import time as t

template_dir = "templates/"
container_dir_detected = "templates/detected"
item_dir = "templates/items/"
items_detected_dir = "templates/items/"
output_file_dir = "results.txt"

sample = cv2.imread(template_dir + 'sample_3.jpg', cv2.IMREAD_GRAYSCALE)
sample1 = cv2.imread(template_dir + 'sample_1.jpg', cv2.IMREAD_GRAYSCALE)
sample2 = cv2.imread(template_dir + 'sample_2.jpg', cv2.IMREAD_GRAYSCALE)
sample3 = cv2.imread(template_dir + 'sample_3.jpg', cv2.IMREAD_GRAYSCALE)
empty_template = cv2.imread(template_dir + "empty_cell_template.jpg", cv2.IMREAD_GRAYSCALE)
empty_w, empty_h = empty_template.shape[::-1]

empty_small_template = cv2.imread("templates/empty_cell_small_template.jpg", cv2.IMREAD_GRAYSCALE)
unknown_item_small_template = cv2.imread("templates/unknown_item_small_template.jpg", cv2.IMREAD_GRAYSCALE)

# Tag - Image - Rows, Columns
container_tags = [("jacket", cv2.imread(template_dir + "jacket_template.png", cv2.IMREAD_GRAYSCALE), (2, 2)),
                  ("pc_block", cv2.imread(template_dir + "pc_template.png", cv2.IMREAD_GRAYSCALE), (4, 4)),
                  ("safe", cv2.imread(template_dir + "safe_template.png", cv2.IMREAD_GRAYSCALE), (3, 3)),
                  ("toolbox", cv2.imread(template_dir + "toolbox_template.png", cv2.IMREAD_GRAYSCALE), (3, 4)),
                  ("technical_supply_crate", cv2.imread(template_dir + "technical_supply_template.png", cv2.IMREAD_GRAYSCALE), (5, 5)),
                  ("duffle_bag", cv2.imread(template_dir + "duffle_template.png", cv2.IMREAD_GRAYSCALE), (3, 4)),
                  ("dead_scav", cv2.imread(template_dir + "dead_scav_template.png", cv2.IMREAD_GRAYSCALE), (4, 4)),
                  ("drawer", cv2.imread(template_dir + "drawer_template.png", cv2.IMREAD_GRAYSCALE), (2, 2)),
                  ("weapon_box", cv2.imread(template_dir + "weapon_box_template.png", cv2.IMREAD_GRAYSCALE), (2, 5))]

# Edge detects and returns the number of pixels found
def scoreFrameDensity(frame, min_grayscale):
    score = 0
    for row in frame:
        for pixel in row:
            if pixel >= min_grayscale:
                score += pixel
    return score

def getClockFromSeconds(time_s):
    h = int(np.floor(time_s / (60*60)))
    m = int(np.floor(time_s / (60))) - h*60
    s = int(np.floor(time_s)) - h*60*60 - m*60
    return str(h) + "h" + str(m) + "m" + str(s) + "s"


#cv2.rectangle(frame, ((w*r), (h*c + 20)), ((w*(r + 1), (h*(c + 1) + 20) - td_h)), (255, 255, 255), 2)
def getCell(cropped_frame, row, col):
    h = 43
    w = 43
    return cropped_frame[(h*col + 20):(h*(col + 1) + 20), (w*row):(w*(row + 1))]

def getFilledCells(frame, rows, cols):
    crop_frame = frame
    good_cells = []
    
    for r in range(rows):
        for c in range(cols):
            cell = getCell(crop_frame, r, c)
            score = scoreFrameDensity(cell, 0)
            if score > 70000:
                good_cells.append((r, c))
    return good_cells

def getTaggedCells(frame, rows, cols):    
    crop_frame = frame
    good_cells = []
    for r in range(rows):
        for c in range(cols):
            cell = getCell(crop_frame, r, c)
            tag = cell[2:10, ::1]
            score = scoreFrameDensity(tag, 100)
            if score > 1000:
                good_cells.append(((r, c), cell))
    return good_cells

def isCellOccupied(frame, row, col):
    cell = getCell(frame, row, col)
    tag = cell[2:10, ::1]
    if scoreFrameDensity(tag, 100) > 6000 or scoreFrameDensity(cell, 0) > 70000:
        return True
    return False

def topLeftCellOccupied(frame):
    if scoreFrameDensity(getCell(frame, 0, 0), 0) > 70000:
        return True
    return False

def getContainerType(frame):
    crop = frame[0:100, 220:500]
    for template in container_tags:
        res = cv2.matchTemplate(crop, template[1], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val >= 0.8:
            return (template[0], template[2])
    random.seed(None)
    name = str(random.randrange(10000)) + "_template.png"
    Image.fromarray(crop).save(container_dir_detected + name)
    print("'" + name + "' created")
    return (name, (0, 0))

def getLootInFrames(loot_items, frames, rows, cols):
    items = loot_items
    loot = []
    
    cooridnates_filled = []
    index = 0
    while (len(cooridnates_filled) == 0) and index < len(frames):
        cooridnates_filled = getFilledCells(frames[index], rows, cols)
        index += 1
    coordinates_logged = []
    
    for frame in frames:
        cells = []
        t_cells = getTaggedCells(frame, rows, cols) # Use this to keep track of which cells come and go based on coordinate position
        for cell in t_cells:
            if not (cell[0] in coordinates_logged) and (cell[0] in cooridnates_filled):
                cells.append(cell)
                coordinates_logged.append(cell[0])
        for cell in cells:
            item_found = False
            for item in items:
                (score, diff) = structural_similarity(cell[1][3:35, 3::], item[0][3:35, 3::], full=True)
                if score > 0.7:
                    item_found = True
                    if not item[1] == "noise":
                        loot.append(item[1])
                    break
            if not item_found:
                uid = str(random.randrange(100000))
                Image.fromarray(cell[1]).save(items_detected_dir + uid + ".png")
                items.append((cell[1], uid))
                loot.append(uid)
    return loot

def loadLootItems():
    items = []
    file_names = [f for f in listdir(item_dir) if isfile(join(item_dir, f))]
    for file_name in file_names:
        name = file_name[0:(len(file_name) - 4)]
        dupe_index = name.find("_dupe")
        if dupe_index > 0:
            name = name[0:dupe_index]
        if "noise" in name:
            name = "noise"
        items.append((cv2.imread(item_dir + file_name, cv2.IMREAD_GRAYSCALE), name))
    return items

# Do not concatenate twice because it changes 'frame_groups' by reference when appending
def concatenateGroups(frame_groups):
    edited_frame_groups = []
    
    prev_group = frame_groups[0]    
    for index in range(1, len(frame_groups)):
        prev_time = prev_group[0]
        prev_frames = prev_group[1]
        prev_last_frame = prev_frames[::-1][0]
        prev_container = getContainerType(prev_last_frame)
        prev_cells = getFilledCells(prev_last_frame, prev_container[1][0], prev_container[1][1])
        
        curr_frames = frame_groups[index][1]
        curr_first_frame = curr_frames[0]
        curr_container = getContainerType(curr_first_frame)
        curr_cells = getFilledCells(curr_first_frame, curr_container[1][0], curr_container[1][1])
        
        find_index = 1
        while len(curr_cells) == 0 and find_index < len(curr_frames):
            curr_first_frame = curr_frames[find_index]
            curr_cells = getFilledCells(curr_first_frame, curr_container[1][0], curr_container[1][1])
            find_index += 1
        
        if prev_container == curr_container and prev_cells == curr_cells:
            print("Merging")
            for frame in curr_frames:
                prev_group[1].append(frame)
        else:
            edited_frame_groups.append((prev_time, prev_container, prev_group[1]))
            if topLeftCellOccupied(curr_first_frame):
                prev_group = frame_groups[index]
            else:
                prev_group = []
    prev_frames = prev_group[1]
    edited_frame_groups.append((prev_group[0], getContainerType(prev_frames[::-1][0]), prev_group[1]))
    
    return edited_frame_groups

def conductAnalysis(frame_groups, streamURL):
    # Load loot items
    loot_items = loadLootItems()
    
    if len(frame_groups) == 0 or len(frame_groups[0]) == 0:
        print("No loot found! Moving on...")
        return []
    
    # Concatenate groups
    edited_frame_groups = concatenateGroups(frame_groups)
    
    loots = []
    f = open(output_file_dir, "a")
    
    # Go through groups, count loot
    for group in edited_frame_groups:
        group_rows = group[1][1][0]
        group_cols = group[1][1][1]
        loot = getLootInFrames(loot_items, group[2], group_rows, group_cols)
        loots.append((group[0], group[1][0], loot))
        f.write("'" + streamURL + "?t=" + str(getClockFromSeconds(group[0]*10)) + "' ~ " + str(group[1][0]) + " ~ " + str(loot) + "\n")
    f.close()
    return loots

def playVideo(final_frames, frame_rate, loot_frames_to_skip):
    display = True
    while display and len(final_frames) > 0:
        for frames in final_frames:
            for frame in frames[1]:
                cv2.imshow('Final Frames', frame)
                if cv2.waitKey(int(1000/frame_rate)*(loot_frames_to_skip + 1)) & 0xFF == ord('s'): 
                    display = False
                    break
            if not display:
                break
    cv2.destroyAllWindows()
    
def playVideoSegment(final_frames_segment, frame_rate, loot_frames_to_skip):
    display = True
    while display and len(final_frames_segment) > 0:
        for frame in final_frames_segment:
            cv2.imshow('Final Frames', frame)
            if cv2.waitKey(int(1000/frame_rate)*(loot_frames_to_skip + 1)) & 0xFF == ord('s'): 
                display = False
                break
        if not display:
            break
    cv2.destroyAllWindows()
