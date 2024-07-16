import numpy as np
import cv2 as cv
import os
import ball_net as bn
import blobber
import sys
from moviepy.editor import *

def draw_ball(mask, frame):
  cnts, _ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

  k = 0
  for c in cnts:
    rx,ry,rw,rh  = cv.boundingRect(c)
    mn = min(rw, rh)
    mx = max(rw, rh)
    r = mx / mn
    if mn < 10 or mx > 40 or r > 1.5:
      continue

    cut_m = mask[ry : ry + rh, rx : rx + rw]
    cut_f = frame[ry : ry + rh, rx : rx + rw]

    cut_c = cv.bitwise_and(cut_f,cut_f,mask = cut_m)
    if bn.check_pic(cut_c) == 0:
      ((x, y), r) = cv.minEnclosingCircle(c)
      cv.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 3)


def test_clip(path):
  vs = cv.VideoCapture(path)
  backSub = cv.createBackgroundSubtractorMOG2()

  n = 0

  frame_rate = round(vs.get(cv.CAP_PROP_FPS))
  frame_width  = vs.get(cv.CAP_PROP_FRAME_WIDTH)
  frame_height = vs.get(cv.CAP_PROP_FRAME_HEIGHT)

  boundaries = []
  frame_window = frame_rate * 2
  ball_count = 0
  prev_was_picked = False
  ratio = 0

  # out = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*"mp4v"), frame_rate, (round(frame_width),round(frame_height)))

  while(True):
    ret, frame = vs.read()

    if not ret or frame is None:
      break

    h = frame.shape[0]
    w = frame.shape[1]
    # curr_frames.append(frame)

    frame = cv.resize(frame, (int(w/2),int(h/2)))
    mask = backSub.apply(frame)

    mask = cv.dilate(mask, None)
    mask = cv.GaussianBlur(mask, (15, 15),0)
    ret,mask = cv.threshold(mask,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    blobber.handle_blobs(mask, frame)

    ball_blob = blobber.get_ball_blob()
    if ball_blob is not None and ball_blob.pts[-1][1] < int(0.5 * 0.5 * frame_height): 
      # print(ball_blob.pts[-1], int(0.4 * 0.5 * frame_height))
      ball_count += 1

    if n != 0 and n % frame_window == 0:
      ratio = ball_count / frame_window
      prev_timestamp = (n - frame_window) / frame_rate
      curr_timestamp = n / frame_rate

      if ratio > 0.15:
        # cv.putText(
        #     frame,
        #     f"Wrote last 2s",
        #     (50, 50),
        #     cv.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (255, 255, 255),
        #     2,
        # )

        # for frame in curr_frames: out.write(frame)

        if not boundaries or boundaries[-1][1] != prev_timestamp:
          boundaries.append([prev_timestamp, curr_timestamp])
        else:
          boundaries[-1][1] = curr_timestamp
        
        prev_was_picked = True
      elif prev_was_picked:
        # cv.putText(
        #     frame,
        #     f"Wrote last 2s",
        #     (50, 50),
        #     cv.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (255, 255, 255),
        #     2,
        # )
        # for frame in curr_frames: out.write(frame)

        if not boundaries or boundaries[-1][1] != prev_timestamp:
          boundaries.append([prev_timestamp, curr_timestamp])
        else:
          boundaries[-1][1] = curr_timestamp

        prev_was_picked = False
      else:
        prev_was_picked = False

      # curr_frames = []
      ball_count = 0
      print(f'At frame {n} or {n // frame_rate} seconds')

    # cv.putText(
    #     frame,
    #     f"{ratio}",
    #     (50, 50),
    #     cv.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (255, 255, 255),
    #     2,
    # )

    # blobber.draw_ball_path(frame)
    # blobber.draw_ball(frame)

    # cv.imwrite("frames/frame-{:03d}.jpg".format(n), frame)
    # cv.imshow('frame', frame)

    if cv.waitKey(10) == 27: break
    n += 1

  if ball_count / (n % frame_window) > 0.2 or prev_was_picked:
      curr_timestamp = (n - (n % frame_window)) / frame_rate
      prev_timestamp = n / frame_rate

      if not boundaries or boundaries[-1][1] != prev_timestamp:
        boundaries.append([prev_timestamp, curr_timestamp])
      else:
        boundaries[-1][1] = curr_timestamp


    # for frame in curr_frames: out.write(frame)

  input = VideoFileClip(path)
  concatenate_videoclips([input.subclip(start,end) for start,end in boundaries]).write_videofile('output.mp4',codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True)

  vs.release()
  # out.release()
  cv.destroyAllWindows()

test_clip(sys.argv[1])
