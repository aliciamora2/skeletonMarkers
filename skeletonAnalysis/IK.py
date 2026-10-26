#!/usr/bin/env python3

import numpy as np
import math
import csv
import time

csv_data = []
input_csv = 'humanPoints.csv' #Path to the csv file to be loaded (e.g. 'data/foto1.csv')
with open(input_csv, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader, None)# Skip the header row
    for row in csv_reader:
        frame_number, landmark, x, y, z = row
        csv_data.append([float(x), float(y), float(z)])

first_column = [row[0] for row in csv_data]

applyJointLimits = False #Flag to apply humanoid joint limits to the final joint angles (True means they are applied)

st = time.time()

J7 = csv_data[7] # Left ear
J8 = csv_data[8] # Right ear
J11 = csv_data[11] # Left shoulder
J12 = csv_data[12] # Right shoulder
J13 = csv_data[13] # Left elbow
J14 = csv_data[14] # Right elbow
J15 = csv_data[15] # Left wrist
J16 = csv_data[16] # Right wrist
J17 = csv_data[17] # Left pinky
J18 = csv_data[18] # Right pinky
J19 = csv_data[19] # Left index
J20 = csv_data[20] # Right index
J23 = csv_data[23] # Left hip
J24 = csv_data[24] # Right hip
J25 = csv_data[25] # Left knee
J26 = csv_data[26] # Right knee
J27 = csv_data[27] # Left ankle
J28 = csv_data[28] # Right ankle
J29 = csv_data[29] # Left heel
J30 = csv_data[30] # Right heel
J31 = csv_data[31] # Left toes
J32 = csv_data[32] # Right toes

a = [(J23[0]+J24[0])/2, (J23[1]+J24[1])/2, (J23[2]+J24[2])/2]
b = [(J11[0]+J12[0])/2, (J11[1]+J12[1])/2, (J11[2]+J12[2])/2]
c = [(J8[0]+J7[0])/2, (J8[1]+J7[1])/2, (J8[2]+J7[2])/2]

vect8_7 = [J7[0]-J8[0], J7[1]-J8[1], J7[2]-J8[2]]
vect11_7 = [J7[0]-J11[0], J7[1]-J11[1], J7[2]-J11[2]]
vect11_8 = [J8[0]-J11[0], J8[1]-J11[1], J8[2]-J11[2]]
vect11_12 = [J12[0]-J11[0], J12[1]-J11[1], J12[2]-J11[2]]
vect11_13 = [J13[0]-J11[0], J13[1]-J11[1], J13[2]-J11[2]]
vect13_15 = [J15[0]-J13[0], J15[1]-J13[1], J15[2]-J13[2]]
vect11_23 = [J23[0]-J11[0], J23[1]-J11[1], J23[2]-J11[2]]
vect12_11 = [J11[0]-J12[0], J11[1]-J12[1], J11[2]-J12[2]]
vect12_14 = [J14[0]-J12[0], J14[1]-J12[1], J14[2]-J12[2]]
vect14_16 = [J16[0]-J14[0], J16[1]-J14[1], J16[2]-J14[2]]
vect12_24 = [J24[0]-J12[0], J24[1]-J12[1], J24[2]-J12[2]]
vect15_17 = [J17[0]-J15[0], J17[1]-J15[1], J17[2]-J15[2]]
vect15_19 = [J19[0]-J15[0], J19[1]-J15[1], J19[2]-J15[2]]
vect16_18 = [J18[0]-J16[0], J18[1]-J16[1], J18[2]-J16[2]]
vect16_20 = [J20[0]-J16[0], J20[1]-J16[1], J20[2]-J16[2]]
vect24_23 = [J23[0]-J24[0], J23[1]-J24[1], J23[2]-J24[2]]
vect23_24 = [J24[0]-J23[0], J24[1]-J23[1], J24[2]-J23[2]]
vect23_25 = [J25[0]-J23[0], J25[1]-J23[1], J25[2]-J23[2]]
vect25_27 = [J27[0]-J25[0], J27[1]-J25[1], J27[2]-J25[2]]
vect24_26 = [J26[0]-J24[0], J26[1]-J24[1], J26[2]-J24[2]]
vect26_28 = [J28[0]-J26[0], J28[1]-J26[1], J28[2]-J26[2]]
vect27_29 = [J29[0]-J27[0], J29[1]-J27[1], J29[2]-J27[2]]
vect27_31 = [J31[0]-J27[0], J31[1]-J27[1], J31[2]-J27[2]]
vect29_31 = [J31[0]-J29[0], J31[1]-J29[1], J31[2]-J29[2]]
vect28_32 = [J32[0]-J28[0], J32[1]-J28[1], J32[2]-J28[2]]
vect28_30 = [J30[0]-J28[0], J30[1]-J28[1], J30[2]-J28[2]]
vect30_32 = [J32[0]-J30[0], J32[1]-J30[1], J32[2]-J30[2]]
vecta_11 = [J11[0]-a[0], J11[1]-a[1], J11[2]-a[2]]
vecta_12 = [J12[0]-a[0], J12[1]-a[1], J12[2]-a[2]]
vecta_b = [b[0]-a[0], b[1]-a[1], b[2]-a[2]]
vectb_a = [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
vectb_c = [c[0]-b[0], c[1]-b[1], c[2]-b[2]]

########################################################### HEAD ###########################################################
#Head roll
HEroll = math.asin(np.dot(vect12_11,vect8_7)/(np.linalg.norm(vect12_11)*np.linalg.norm(vect8_7)))
HEroll = math.degrees(HEroll) - 90
if np.linalg.norm(np.cross(vect11_7,vect11_12))>np.linalg.norm(np.cross(vect11_8,vect11_12)):
    HEroll = -HEroll

#Head pitch
Nvect11_12_a = np.cross(vecta_11,vecta_12)
HEpitch = math.asin(np.dot(Nvect11_12_a,vectb_c)/(np.linalg.norm(Nvect11_12_a)*np.linalg.norm(vectb_c)))
HEpitch = -math.degrees(HEpitch)
if np.dot(Nvect11_12_a,vectb_c)>0:
    HEpitch = -HEpitch

#Head yaw
HEyaw = math.asin(np.dot(Nvect11_12_a,vect8_7)/(np.linalg.norm(Nvect11_12_a)*np.linalg.norm(vect8_7)))
HEyaw = -math.degrees(HEyaw)

########################################################## TORSO ###########################################################
#Torso roll
Troll = math.asin(np.dot(vecta_b,vect12_11)/(np.linalg.norm(vecta_b)*np.linalg.norm(vect12_11)))
Troll = math.degrees(Troll)

#Torso yaw
N1 = np.cross(vecta_b,vect23_24)
N2 = np.cross(vecta_b,vect11_12)
Tyaw = math.acos(np.dot(N1,N2)/(np.linalg.norm(N1)*np.linalg.norm(N2)))
Tyaw = math.degrees(Tyaw)
if np.dot(N1,vect11_12)<0:
    Tyaw = -Tyaw

#Torso pitch
Tpitch = math.asin(np.dot([0,-1,0],N2)/(np.linalg.norm([0,-1,0])*np.linalg.norm(N2)))
Tpitch = -math.degrees(Tpitch)

######################################################### LEFT ARM #########################################################
#---------------------------------- Shoulder ----------------------------------#
#Shoulder roll
LSroll = math.asin(np.dot(vect12_11,vect11_13)/(np.linalg.norm(vect12_11)*np.linalg.norm(vect11_13)))
LSroll = math.degrees(LSroll)

#Shoulder pitch
Nvect11_12_23 = np.cross(vect11_12,vectb_a)
LSpitch = math.asin(np.dot(vect11_13,Nvect11_12_23)/(np.linalg.norm(vect11_13)*np.linalg.norm(Nvect11_12_23)))
NvectPerp11_12_23 = np.cross(Nvect11_12_23,vect11_12)
pitchOrientation = np.dot(vect11_13,NvectPerp11_12_23)
if pitchOrientation<0:
    if math.degrees(LSpitch)>=0:
        LSpitch = 180 - math.degrees(LSpitch)
    else:
        LSpitch = -180 - math.degrees(LSpitch)
else:
    LSpitch = math.degrees(LSpitch)
LSpitch = -LSpitch

#Shoulder yaw
Nvect11_12_13 = np.cross(vect11_12,vect11_13)
NvectPerp11_12_13 = np.cross(Nvect11_12_13,vect11_13)
LSyaw = math.asin(np.dot(vect13_15,NvectPerp11_12_13)/(np.linalg.norm(vect13_15)*np.linalg.norm(NvectPerp11_12_13)))
LSyaw = math.degrees(LSyaw)

#------------------------------ Elbow and wrist ------------------------------#
#Elbow pitch
LEpitch = -np.arccos(np.dot(vect11_13,vect13_15)/(np.linalg.norm(vect11_13)*np.linalg.norm(vect13_15)))
LEpitch = math.degrees(LEpitch)
#Elbow yaw
Nvect15_17_19 = np.cross(vect15_17,vect15_19)
Lperp = np.cross(Nvect15_17_19,vect13_15)
Nvect11_13_15 = np.cross(vect11_13,vect13_15)
LEyaw = math.asin(np.dot(Lperp,Nvect11_13_15)/(np.linalg.norm(Lperp)*np.linalg.norm(Nvect11_13_15)))
LEyaw = math.degrees(-LEyaw)

#Wrist roll
LNroll = np.cross(vect15_17,vect15_19)
LWroll = np.arcsin(np.dot(vect13_15,LNroll)/(np.linalg.norm(vect13_15)*np.linalg.norm(LNroll)))
LWroll = math.degrees(LWroll)

######################################################### RIGHT ARM #########################################################
#---------------------------------- Shoulder ----------------------------------#
#Shoulder roll
RSroll = math.asin(np.dot(vect11_12,vect12_14)/(np.linalg.norm(vect11_12)*np.linalg.norm(vect12_14)))
RSroll = -math.degrees(RSroll)

#Shoulder pitch
Nvect12_24_11 = np.cross(vectb_a,vect12_11)
RSpitch = math.asin(np.dot(vect12_14,Nvect12_24_11)/(np.linalg.norm(vect12_14)*np.linalg.norm(Nvect12_24_11)))
NvectPerp12_24_11 = np.cross(Nvect12_24_11,vect12_11)
pitchOrientation = np.dot(vect12_14,NvectPerp12_24_11)
if pitchOrientation>0:
    if math.degrees(RSpitch)>=0:
        RSpitch = 180 - math.degrees(RSpitch)
    else:
        RSpitch = -180 - math.degrees(RSpitch)
else:
    RSpitch = math.degrees(RSpitch)
RSpitch = -RSpitch

#Shoulder yaw
Nvect12_14_11 = np.cross(vect12_14,vect12_11)
NvectPerp12_14_11 = np.cross(Nvect12_14_11,vect12_14)
RSyaw = math.asin(np.dot(vect14_16,NvectPerp12_14_11)/(np.linalg.norm(vect14_16)*np.linalg.norm(NvectPerp12_14_11)))
RSyaw = math.degrees(RSyaw)

#------------------------------ Elbow and wrist ------------------------------#
#Elbow pitch
REpitch = -np.arccos(np.dot(vect12_14,vect14_16)/(np.linalg.norm(vect12_14)*np.linalg.norm(vect14_16)))
REpitch = math.degrees(REpitch)

#Elbow yaw
Nvect16_20_18 = np.cross(vect16_20,vect16_18)
Rperp = np.cross(Nvect16_20_18,vect14_16)
Nvect12_16_14 = np.cross(vect14_16,vect12_14)
REyaw = np.arcsin(np.dot(Rperp,Nvect12_16_14)/(np.linalg.norm(Rperp)*np.linalg.norm(Nvect12_16_14)))
REyaw = math.degrees(-REyaw)

#Wrist roll
RNroll = np.cross(vect16_20,vect16_18)
RWroll = np.arcsin(np.dot(vect14_16,RNroll)/(np.linalg.norm(vect14_16)*np.linalg.norm(RNroll)))
RWroll = math.degrees(-RWroll) 

######################################################### LEFT LEG #########################################################
#----------------------------------- Waist ------------------------------------#
#Waist roll
LWroll = np.arcsin(np.dot(vect24_23,vect23_25)/(np.linalg.norm(vect24_23)*np.linalg.norm(vect23_25)))
LWroll = -math.degrees(LWroll)

#Waist pitch
waistPlane = np.cross([0,1,0],vect24_23)
LWpitch = math.asin(np.dot(waistPlane,vect23_25)/(np.linalg.norm(waistPlane)*np.linalg.norm(vect23_25)))
LWpitch = -math.degrees(LWpitch)

#Waist yaw
Nvect24_23_25 = np.cross(vect23_24,vect23_25)
NvectPerp24_23_25 = np.cross(Nvect24_23_25,vect23_25)
LWyaw = math.asin(np.dot(vect25_27,NvectPerp24_23_25)/(np.linalg.norm(vect25_27)*np.linalg.norm(NvectPerp24_23_25)))
LWyaw = -math.degrees(LWyaw)

#----------------------------------- Knee -------------------------------------#
#Knee pitch
LKpitch = np.arccos(np.dot(vect23_25,vect25_27)/(np.linalg.norm(vect23_25)*np.linalg.norm(vect25_27)))
LKpitch = math.degrees(LKpitch)

#----------------------------------- Ankle ------------------------------------#
#Ankle roll
LanklePlane = np.cross(vect27_29,vect27_31)
LAroll = math.asin(np.dot(LanklePlane,vect25_27)/(np.linalg.norm(LanklePlane)*np.linalg.norm(vect25_27)))
LAroll = math.degrees(LAroll)

#Ankle pitch
LApitch = math.asin(np.dot(vect29_31,vect25_27)/(np.linalg.norm(vect29_31)*np.linalg.norm(vect25_27)))
LApitch = math.degrees(LApitch)

######################################################## RIGHT LEG #########################################################
#----------------------------------- Waist ------------------------------------#
#Waist roll
RWroll = np.arcsin(np.dot(vect23_24,vect24_26)/(np.linalg.norm(vect23_24)*np.linalg.norm(vect24_26)))
RWroll = -math.degrees(RWroll)

#Waist pitch
RWpitch = math.asin(np.dot(waistPlane,vect24_26)/(np.linalg.norm(waistPlane)*np.linalg.norm(vect24_26)))
RWpitch = -math.degrees(RWpitch)

#Waist yaw
Nvect24_26_23 = np.cross(vect24_26,vect24_23)
NvectPerp24_26_23 = np.cross(Nvect24_26_23,vect24_26)
RWyaw = math.asin(np.dot(vect26_28,NvectPerp24_26_23)/(np.linalg.norm(vect26_28)*np.linalg.norm(NvectPerp24_26_23)))
RWyaw = -math.degrees(RWyaw)

#----------------------------------- Knee -------------------------------------#
#Knee pitch
RKpitch = np.arccos(np.dot(vect24_26,vect26_28)/(np.linalg.norm(vect24_26)*np.linalg.norm(vect26_28)))
RKpitch = math.degrees(RKpitch)

#----------------------------------- Ankle ------------------------------------#
#Ankle roll
RanklePlane = np.cross(vect28_30,vect30_32)
RAroll = math.asin(np.dot(RanklePlane,vect26_28)/(np.linalg.norm(RanklePlane)*np.linalg.norm(vect26_28)))
RAroll = math.degrees(RAroll)

#Ankle pitch
RApitch = math.asin(np.dot(vect30_32,vect26_28)/(np.linalg.norm(vect30_32)*np.linalg.norm(vect26_28)))
RApitch = math.degrees(RApitch)

et = time.time()
# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

###################################################### JOINT LIMITS ########################################################
if applyJointLimits:
    LWyaw = max(min(LWyaw, 45.0), -45.0)
    LWroll = max(min(LWroll, 25.0), -25.0)
    LWpitch = max(min(LWpitch, 25.0), -77.0)
    LKpitch = max(min(LKpitch, 125.0), -1.0)
    LApitch = max(min(LApitch, 16.0), -57.0)
    LAroll = max(min(LAroll, 34.0), -34.0)
    RWyaw = max(min(RWyaw, 45.0), -45.0)
    RWroll = max(min(RWroll, 25.0), -25.0)
    RWpitch = max(min(RWpitch, 25.0), -77.0)
    RKpitch = max(min(RKpitch, 125.0), -1.0)
    RApitch = max(min(RApitch, 16.0), -57.0)
    RAroll = max(min(RAroll, 34.0), -34.0)
    Tpitch = max(min(Tpitch, 30.0), -12.0)
    Troll = max(min(Troll, 25.0), -25.0)
    Tyaw = max(min(Tyaw, 35.0), -35.0)
    HEyaw = max(min(HEyaw, 70.0), -70.0)
    HEroll = max(min(HEroll, 23.0), -23.0)
    HEpitch = max(min(HEpitch, 25.0), -30.0)
    LSpitch = max(min(LSpitch, 60.0), -147.0)
    LSroll = max(min(LSroll, 89.0), -10.0)
    LSyaw = max(min(LSyaw, 92.0), -92.0)
    LEpitch = max(min(LEpitch, 2.0), -123.0)
    LEyaw = max(min(LEyaw, 92.0), -92.0)
    LWroll = max(min(LWroll, 84.0), -84.0)
    RSpitch = max(min(RSpitch, 60.0), -147.0)
    RSroll = max(min(RSroll, 10.0), -89.0)
    RSyaw = max(min(RSyaw, 92.0), -92.0)
    REpitch = max(min(REpitch, 2.0), -123.0)
    REyaw = max(min(REyaw, 92.0), -92.0)
    RWroll = max(min(RWroll, 84.0), -84.0)

####################################################### PRINT INFO. ########################################################
print("Left waist orientation is (0,1,2): %s, %s, %s" % (round(LWyaw,1), round(LWroll,1), round(LWpitch,1)))
print("Left knee orientation is (3): %s" % round(LKpitch,1))
print("Left ankle orientation is (4,5): %s, %s" % (round(LApitch,1), round(LAroll,1)))
print("Right waist orientation is (7,8,9): %s, %s, %s" % (round(RWyaw,1), round(RWroll,1), round(RWpitch,1)))
print("Right knee orientation is (10): %s" % round(RKpitch,1))
print("Right ankle orientation is (11,12): %s, %s" % (round(RApitch,1), round(RAroll,1)))
print("Torso orientation is (14,15,16): %s, %s, %s" % (round(Tpitch,1), round(Troll,1), round(Tyaw,1)))
print("Head orientation is (17,18,19): %s, %s, %s" % (round(HEyaw,1), round(HEroll,1), round(HEpitch,1)))
print("Left shoulder orientation is (20,21,22): %s, %s, %s" % (round(LSpitch,1), round(LSroll,1), round(LSyaw,1)))
print("Left elbow orientation is (23,24): %s, %s" % (round(LEpitch,1), round(LEyaw,1)))
print("Left wrist orientation is (25): %s" % round(LWroll,1))
print("Right shoulder orientation is (rpy)(26,27,28): %s, %s, %s" % (round(RSpitch,1), round(RSroll,1), round(RSyaw,1)))
print("Right elbow orientation is (p,y)(29,30): %s, %s" % (round(REpitch,1), round(REyaw,1)))
print("Right wrist orientation is (r)(31): %s" % round(RWroll,1))
