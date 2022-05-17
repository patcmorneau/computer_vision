# A comparative analysis of SIFT, SURF, KAZE, AKAZE, ORB, and BRISK
# https://ieeexplore.ieee.org/document/8346440
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

MAX_FEATURES = 50000
GOOD_MATCH_PERCENT = 0.15

# As of Python version 3.7, dictionaries are ordered. In Python 3.6 and earlier, dictionaries are unordered.
execTime = {}

im1 = cv2.imread("images/1.jpg", cv2.IMREAD_COLOR)
im2 = cv2.imread("images/2.jpg", cv2.IMREAD_COLOR)

# Convert images to grayscale
im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Detect ORB features and compute descriptors.
orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

print("number of keypoints: ", len(descriptors1))
print("number of keypoints: ", len(descriptors2))

"""
print(descriptors1)
print(len(descriptors1))
print(len(descriptors1[0]))
"""

"""
im1Keypoints = np.array([])
im1Keypoints = cv2.drawKeypoints(im1, keypoints1, im1Keypoints, color=(0,0,255),flags=0)
print("Saving Image with Keypoints")
cv2.imwrite("keypoints.jpg", im1Keypoints)
"""


print("BRUTEFORCE_HAMMING")
start = time.time_ns()
# Match features.
matcher = cv2.DescriptorMatcher_create(
                cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)

matches = list(matches)
# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]
end = time.time_ns()
execTime[end-start] = "BRUTEFORCE_HAMMING"
print("time", end - start)
print("matches: ",len(matches))


"""
TODO:
print("FLANNBASED")
start = time.time_ns()
# Match features.
matcher = cv2.DescriptorMatcher_create(
                cv2.DESCRIPTOR_MATCHER_FLANNBASED)
matches = matcher.match(descriptors1, descriptors2, None)

matches = list(matches)
# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]
end = time.time_ns()
execTime.append(end-start)
print("time", end - start)
print("matches: ",len(matches))
"""

print("BRUTEFORCE_HAMMINGLUT")
start = time.time_ns()
# Match features.
matcher = cv2.DescriptorMatcher_create(
                cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT)
matches = matcher.match(descriptors1, descriptors2, None)

matches = list(matches)
# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]
end = time.time_ns()
execTime[end-start] = "BRUTEFORCE_HAMMINGLUT"
print("time", end - start)
print("matches: ",len(matches))


print("BRUTEFORCE")
start = time.time_ns()
# Match features.
matcher = cv2.DescriptorMatcher_create(
                cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
matches = matcher.match(descriptors1, descriptors2, None)

matches = list(matches)
# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]
end = time.time_ns()
execTime[end-start] = "BRUTEFORCE"
print("time", end - start)
print("matches: ",len(matches))


print("BRUTEFORCE_L1")
start = time.time_ns()
# Match features.
matcher = cv2.DescriptorMatcher_create(
                cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)
matches = matcher.match(descriptors1, descriptors2, None)

matches = list(matches)
# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]
end = time.time_ns()
execTime[end-start] = "BRUTEFORCE_L1"
print("time", end - start)
print("matches: ",len(matches))


print("BRUTEFORCE_SL2")
start = time.time_ns()
# Match features.
matcher = cv2.DescriptorMatcher_create(
                cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_SL2)
matches = matcher.match(descriptors1, descriptors2, None)

matches = list(matches)
# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]
end = time.time_ns()
execTime[end-start] = "BRUTEFORCE_SL2"
print("time", end - start)
print("matches: ",len(matches))

print("\nAnd the big big big winner is:")
print(list(execTime.values())[0])
