import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import pywavefront
pygame.init()
screen = pygame.display.set_mode((600, 400))
cap = cv2.imread('C:\\Mario.png')
model = cv2.imread('C:\\mario 3d.jpg', 0)
fox1 = pywavefront.Wavefront('low-poly-fox-by-pixelmannen.obj', collect_faces=True)
cap = np.array(cap)
model = np.array(model)
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(cap, None)
kp2, des2 = orb.detectAndCompute(model, None)
plt.imshow(cap), plt.show()
plt.imshow(model), plt.show()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(cap, kp1, model, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3), plt.show()
# assuming matches stores the matches found and
# returned by bf.match(des_model, des_frame)
# differenciate between source points and destination points
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
# compute Homography
homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# Draw a rectangle that marks the found model in the frame
h, w = model.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
# project corners into frame
dst = cv2.perspectiveTransform(pts, homography)
# connect them with lines
img2 = cv2.polylines(model, [np.int32(dst)], True, 150, 3, cv2.LINE_AA)
cv2.imshow('frame', model)
cv2.waitKey(0)
global camera
camera = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
def projection_matrix(homography, camera):
 projection_matrix(homography, camera)
inv_camera = np.linalg.inv(camera)
rot_and_transl = inv_camera.dot(homography)
col_1 = rot_and_transl[:, 0]
col_2 = rot_and_transl[:, 1]
col_3 = rot_and_transl[:, 2]
 # normalise vectors
l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
rot_1 = col_1 / l
rot_2 = col_2 / l
translation = col_3 / l
# compute the orthonormal basis
c = rot_1 + rot_2
p = np.cross(rot_1, rot_2)
d = np.cross(c, p)
rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
rot_3 = np.cross(rot_1, rot_2)
# finally, compute the 3D projection matrix from the model to the current frame
projection = np.stack((rot_1, rot_2, rot_3, translation)).T
# def render(model, fox1, projection, cap, color=False):
scene_box = (fox1.vertices[0], fox1.vertices[0])
for vertex in fox1.vertices:
   min_v = [min(scene_box[0][i], vertex[i]) for i in range(3)]
   max_v = [max(scene_box[1][i], vertex[i]) for i in range(3)]
scene_box = (min_v, max_v)
scene_size = [scene_box[1][i]-scene_box[0][i] for i in range(3)]
max_scene_size = max(scene_size)
scaled_size = 5
scene_scale = [scaled_size/max_scene_size for i in range(3)]
scene_trans = [-(scene_box[1][i]+scene_box[0][i])/2 for i in range(3)]










def model():
# glPushMatrix()
# glScalef(*scene_scale)
# glTranslatef(*scene_trans)

 for mesh in fox1.mesh_list:
  glBegin(GL_TRIANGLES)
  for face in mesh.faces:
   for vertex_i in face:
    glVertex3f(*fox1.vertices[vertex_i])
#    glEnd()

    glPopMatrix()


def main():

   pygame.init()
   display = (800, 600)
   pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
   gluPerspective(45, (display[0] / display[1]), 1, 500.0)
   glTranslatef(0.0, 0.0, -10)

while True:
 pygame.init()
 for event in pygame.event.get():
     if event.type == pygame.QUIT:
       pygame.quit()
quit()
# if event.type == pygame.KEYDOWN:
 #   if event.key == pygame.K_LEFT:
 #     glTranslatef(-0.5,0,0)
 #     if event.key == pygame.K_RIGHT:
 #      glTranslatef(0.5,0,0)
 #      if event.key == pygame.K_UP:
 #       glTranslatef(0,1,0)
 #       if event.key == pygame.K_DOWN:
 #         glTranslatef(0,-1,0)


# glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
# glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
model()
glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
# pygame.display.flip()
pygame.display.update()
pygame.time.wait(10)

main()
points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
imgpts = np.int32(dst)
if color is False:
 cv2.fillConvexPoly(model, imgpts, (137, 27, 211))
else:
 color = hex_to_rgb(face[-1])
 color = color[::-1]  # reverse
 cv2.fillConvexPoly(model, imgpts, color)
