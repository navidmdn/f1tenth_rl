from __future__ import print_function
from codecs import encode
from pickle import NONE
# import sys
import math
# from typing import ByteString
import numpy as np
from numpy.lib.function_base import _gradient_dispatcher

# import tf2_py
# from tf.transformations import quaternion_from_euler
# from visualization_msgs.msg import Marker
# from geometry_msgs.msg import Point,TransformStamped,PointStamped, PoseStamped
# from std_msgs.msg import ColorRGBA
# from tf2_ros import TransformListener, Buffer
# from tf2_geometry_msgs import do_transform_point


#ROS Imports
# import rospy
# from sensor_msgs.msg import Image, LaserScan
# from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


VELOCITY_X = "linear_vel_x"
VELOCITY_Y = "linear_vel_y"
ANGULAR_VEL_Z = "angular_vel_z"
POSE_X = "pose_x"
POSE_Y = "pose_y"
POSE_THETA = "pose_theta"


# Params
RADIUS = 8 #m  radius for sector
MAX_SECTOR_SIZE = 4 # 4 # original 6
MIN_R_GAP = 3.5
ANGLE_START = 180
KP = 1.8 # 1.8 best for both
R_car = 0.15
VISUALIZATION = False

ANGLE_MIN = -1*math.pi
ANGLE_MAX = math.pi
ANGLE_INCREMENT = 2*math.pi/1080
LEN_RANGES = 1080
ANGLE_END = LEN_RANGES - ANGLE_START - 1

Kp=0.3
# R_car = 0.11
R_car = 0.19
#
# RED = ColorRGBA(1,0,0,1)
# GREEN = ColorRGBA(0,1,0,1)
# BLUE = ColorRGBA(0,0,1,1)
# NO_COLOR= ColorRGBA(0,0,0,0)
# ORIGIN = Point(0,0,0)


# Acceleration controls
BUFFER_DIST = 8 # 8
ACCELERATION = 9
DECELERATION = 12 # 8
FINAL_V = 8




class Vector:
    x : float =0
    y : float =0
    theta : float =0

    def __init__(self,name):
        self.name = name

    def set(self,x,y,theta=0):
        self.x = x
        self.y = y
        self.theta = theta

    def magnitude(self):
        return math.sqrt(self.x*self.x + self.y*self.y)

    
    def __str__(self) -> str:
        return self.name + "\nx = "+str(self.x)+"\ny = "+str(self.y)+"\ntheta = "+str(self.theta)+"\n"
    
    __repr__ = __str__

class Point:
    x = None
    y = None

class Sector:
    # sec_r = RADIUS
    sec_theta = 0
    start_angle = 0
    #ranges = []
    max_dis_to_go = 100
    cost = None
    blocked = False

    padded_at = 100

    def __init__(self,start_angle:float, sec_r=RADIUS):
        self.sec_r = sec_r
        self.start_angle = start_angle
        self.ranges = []

    def cost(self, cp):
        ci = self.centre_of_sector()
        a = abs(cp-ci)
        b = abs(self.centre_of_sector())
        c = self.sector_max_dis()
        return 2*a+2*b + 5/c

    def cost_by_dis(self):
        abs_ = abs(self.centre_of_sector())
        return (0.5*self.sector_max_dis() - abs_  )*-1
        # return 1/self.max_dis_to_go

    def get_point_viz(self):
        theta = self.centre_of_sector()
        pt = Point()
        pt.x = self.sector_max_dis()*math.cos(theta)
        pt.y = self.sector_max_dis()*math.sin(theta)
        return pt

    def sector_max_dis(self):
        return min(self.padded_at,self.max_dis_to_go)

    def get_point_start_sector(self):
        pt = Point()
        pt.x = self.sector_max_dis()*math.cos(self.start_angle)
        pt.y = self.sector_max_dis()*math.sin(self.start_angle)
        return pt

    def get_point_end_sector(self):
        pt = Point()
        pt.x = self.sector_max_dis()*math.cos(self.start_angle+self.sec_theta)
        pt.y = self.sector_max_dis()*math.sin(self.start_angle+self.sec_theta)
        return pt


    def add_range(self,r,theta):
        self.ranges.append(r)
        # print(self.ranges)
        self.sec_theta = abs(self.start_angle-theta)
        if r < self.max_dis_to_go:
            self.max_dis_to_go = r
        if self.sector_max_dis() < MIN_R_GAP :
            self.blocked = True
        # return

    def change_max_dis(self,dis:float):
        if self.max_dis_to_go > dis :
            self.max_dis_to_go = dis
        return
    
    def get_theta_pad_radius(self):
        return math.asin(R_car/self.max_dis_to_go)
    
    def padding_dis(self):
        return self.max_dis_to_go - R_car
    
    def sectors_to_pad(self):
        n = self.get_theta_pad_radius()//6*ANGLE_INCREMENT
        n = int(n)
        n +=1
        return n

    def pad_sector(self,d:int):
        if self.max_dis_to_go > d and self.padded_at > d:
            self.padded_at = d
        if self.sector_max_dis() < MIN_R_GAP :
            self.blocked = True

    def centre_of_sector(self):
        return self.start_angle + self.sec_theta/2.0

    def print(self):
        s = "Sector:\n ("+str(self.start_angle)+", "+str(self.start_angle+self.sec_theta)+")"
        s+= "\n"+str(self.ranges)+"\n"
        #rospy.loginfo(s)

    def __str__(self):
        s = "Sector:\n ("+str(self.start_angle)+", "+str(self.start_angle+self.sec_theta)+")"
        s+= "\n"+str(self.ranges)+"\n"
        # s+=
        return s 

    __repr__ = __str__





class Gap:
    start = None

    def __init__(self):
        self.length= 0
        self.sectors = []
    
    def add_sector(self,sector:Sector):
        self.sectors.append(sector)
        if self.start == None :
            self.start = sector
        self.end= sector
        self.length+=1

    def score(self):
        #f = x*self.length + a* self.mid().centre_of_sector()
        # return self.length
        if self.length == 0:
            return -360
        return -1*abs(self.mid().centre_of_sector())

    def gaps_theta(self,theta:float)->int:
        alpha = self.start.sec_theta
        return abs(int((theta//alpha)+1))

    def mid(self)->Sector:
        if self.length == 0:
            print(self.length)
        return self.sectors[self.length//2]

class VFH:

    #trans: TransformStamped = None
    #CP = 0.0

    def __init__(self):
        # lidarscan_topic = '/scan'
        # drive_topic = '/nav'
        self.sectors: list[Sector] = []

        self.velocity = Vector("velocity")
        self.pose = Vector("pose")

        # tf2
        #self.buffer = Buffer()
        #self.listener = TransformListener(self.buffer)
        if VISUALIZATION:
            self.visualizer = Visualizer()


    # def transform_range(self,r:float,theta:float) -> Point: 
    #     lidar_pt:PointStamped = PointStamped()
    #     if self.trans == None:
    #         self.trans:TransformStamped = self.buffer.lookup_transform("base_link","laser",rospy.Time(0))
    #     lidar_pt.point.x = r * math.cos(theta)
    #     lidar_pt.point.y = r * math.sin(theta)
    #     base_pt = do_transform_point(lidar_pt,self.trans)
    #     x = base_pt.point.x
    #     y = base_pt.point.y
    #     return (math.sqrt(x*x+y*y),math.atan(y/x))




    def process_observation(self,ranges,ego_odom:dict):
        r = ego_odom.get(VELOCITY_X,0)
        theta = ego_odom.get(ANGULAR_VEL_Z,0)
        self.velocity.set(r*math.cos(theta),r*math.sin(theta))
        # self.pose.set(get(obs,POSE_X),get(obs,POSE_Y),get(obs,POSE_THETA))
        self.sectors = []
        angle = ANGLE_MIN + ANGLE_INCREMENT*ANGLE_START
        added = 0
        #for ind in range(300,779):
        for ind in range(ANGLE_START,ANGLE_END):
            r = ranges[ind]
            if r > RADIUS:
               r = RADIUS
            # TODO : 
            # if r == math.nan:
            #     print("zeroo")
            # r_transformed, theta_transformed = self.transform_range(r,angle)
            r_transformed = r
            theta_transformed = angle
            angle += ANGLE_INCREMENT
            
            if added == 0 :
                sector = Sector(theta_transformed)
                self.sectors.append(sector)
            sector.add_range(r_transformed,theta_transformed)
            added+=1
            if added >= MAX_SECTOR_SIZE:
                added = 0
        # padding

        self.pad_sectors()
        best = self.get_best_sector()
        
        front = ranges[LEN_RANGES//2]
        angle = self.get_angle_from_sector(best)
        speed = self.get_drive(angle)

        # print(velocity)
        accelerate , distance = self.to_accelerate(self.velocity.magnitude(),front)

        if accelerate and abs(angle) < math.radians(0.5):
            # print("accelerated")
            speed = 10 # 17 #15,16

        if  front < 1: 
            speed = 0
        
        # visualization
        # if VISUALIZATION:
        #     self.visualizer.publish_pose(pose)
        #     self.visualizer.publish_padding(self.sectors)
        #     self.visualizer.publish_best_sector_dir(best)
        #     self.visualizer.publish_velocity(velocity)
        
        # if velocity.magnitude() > 9:
        #     print(velocity.magnitude()) 
        return speed, angle



    def to_accelerate2(self,v,s):
        d = -2 - 0.03*v*v + s/2 # -2 with 15 v kp 2
        if d > 0:
            return True , d
        return False, d

    def to_accelerate(self,v,s):
        num = (FINAL_V * FINAL_V) - (v*v) + (2 * ACCELERATION * (s - BUFFER_DIST))
        deno = 2 * (ACCELERATION + DECELERATION)
        d = num / deno 
        #d = -2 - 0.03*v*v + s/2 # -2 with 15 v kp 2
        if d > 0:
            return True , d
        return False , d


    def get_drive(self,angle):
        velocity = 5 # 9
        abs_angle = abs(math.degrees(angle))
        if abs_angle > 40:
            velocity = 1.1 #velocity/8
        elif abs_angle > 20 :
            velocity =  4.5 #velocity/2
            # angle += 0.
        elif abs_angle >  10:
            velocity =  6.25 #5*velocity/8
        #velocity *=1/2
        return velocity

    def get_angle_from_sector(self,best:Sector):
        r = best.padded_at
        theta = best.centre_of_sector()
        x = r*math.cos(theta)
        y =r*math.sin(theta)
        LOOK_AHEAD = 0.2
        x_d = x #- LOOK_AHEAD
        y_d = y 
        L_d_square = x_d*x_d + y_d*y_d
        gamma = 2*y_d/(L_d_square)
        # 0.25 almost crash
        Kp = 2 # 1.8 speed 9 #1.5 speed 10 # 1 speed 8 #0.9 # 0.50 earlier
        angle = KP*gamma

        #angle = best.centre_of_sector()
        return angle


    def pad_sectors(self):
        for ind in range(0,len(self.sectors)):
            sec = self.sectors[ind]
            alpha = sec.get_theta_pad_radius()
            pad_at = sec.padding_dis()
            self.pad_from_sector(ind,alpha,sec.centre_of_sector(),pad_at)
            
    def pad_from_sector(self,index:int, alpha:float, theta:float,pad_at: float):
        blocked_ahead = False
        blocked_behind = False
        self.sectors[index].pad_sector(pad_at)
        # ahead = 1
        # behind = 1
        add_ind = 1
        while not blocked_ahead or not blocked_behind:
            if index+add_ind >= len(self.sectors):
                blocked_ahead = True
            else:
                sec_ahead = self.sectors[index+add_ind]

            if index-add_ind < 0:
                blocked_behind = True
            else:
                sec_behind = self.sectors[index - add_ind]


            if not blocked_ahead and theta -alpha <= sec_ahead.start_angle <= theta+alpha:
                sec_ahead.pad_sector(pad_at)
            else:
                blocked_ahead = True
    
            if not blocked_behind and theta-alpha <= sec_behind.start_angle+sec_behind.sec_theta <= theta + alpha:
                sec_behind.pad_sector(pad_at)
            else:
                blocked_behind = True
            add_ind += 1



    def get_best_sector(self)->Sector:
        return self.get_best_sector_by_gap()
        # return self.get_best_sector_by_cost()

    def get_best_sector_by_cost(self)-> Sector:
        sector = self.sectors[0]
        min_cost = sector.cost(self.CP) 
        for s in self.sectors:
            if s.cost(self.CP) < min_cost:
                min_cost = s.cost(self.CP)
                sector = s
        return sector

    def get_best_sector_by_gap(self)->Sector:
        gap = Gap()
        best_gap = gap
        for sector in self.sectors:
            if not sector.blocked:
                gap.add_sector(sector)
            else:
                if best_gap.score() < gap.score():
                    best_gap = gap
                gap = Gap()
                #print(gap.sectors)
        if best_gap.length < 1:
            sec =  self.sectors[len(self.sectors)//2]
            print("fall back sector centre = ",sec.centre_of_sector())
            return sec
        return best_gap.mid()



#
# class Visualizer:
#
#     def __init__(self) -> None:
#         rospy.init_node("VFH_node", anonymous=True)
#
#         self.viz_pub = rospy.Publisher("/vfh_viz",Marker,queue_size=10)
#         self.pose_pub = rospy.Publisher("/pose",PoseStamped,queue_size=10)
#
#         self.pose_stanped = PoseStamped()
#         self.pose_stanped.header.frame_id = "/map"
#
#
#         self.pad_marker = self.init_marker(0,"pad_marker",Marker.LINE_LIST,RED)
#
#         self.best_sec_marker = self.init_marker(1,"best_sec_marker",Marker.ARROW,RED)
#         self.velocity_marker = self.init_marker(2,"velocity",Marker.ARROW,BLUE)
#
#
#
#     def init_marker(self, id, name_space,type_marker,color:ColorRGBA,duration = 0.05,action=Marker.ADD)->Marker:
#         marker = Marker()
#         marker.header.frame_id = "laser"
#         marker.ns = name_space
#         marker.type = type_marker
#
#         marker.lifetime = rospy.Duration(duration)
#         marker.action = Marker.ADD
#         marker.pose.position.z = 0
#         marker.scale.x = 0.05
#         marker.scale.y = 0.05
#         marker.scale.z = 0.05
#         marker.color = color
#         marker.id = id
#         return marker
#
#     def publish_pose(self,pose:Vector):
#         self.pose_stanped.header.stamp = rospy.Time.now()
#         self.pose_stanped.pose.position.x = pose.x
#         self.pose_stanped.pose.position.y = pose.y
#
#         qu = quaternion_from_euler(0,0,pose.theta)
#
#         self.pose_stanped.pose.orientation.x = qu[0]
#         self.pose_stanped.pose.orientation.y = qu[1]
#         self.pose_stanped.pose.orientation.z = qu[2]
#         self.pose_stanped.pose.orientation.w = qu[3]
#         self.pose_pub.publish(self.pose_stanped)
#
#
#     def publish_padding(self, sectors):
#         self.pad_marker.points = []
#         self.pad_marker.colors = []
#         for sec in sectors :
#             self.pad_marker.points.append(sec.get_point_start_sector())
#             self.pad_marker.points.append(sec.get_point_end_sector())
#             self.pad_marker.colors.append(RED)
#             self.pad_marker.colors.append(BLUE)
#         self.viz_pub.publish(self.pad_marker)
#
#
#     def publish_best_sector_dir(self,sector:Sector):
#         self.best_sec_marker.points = [ORIGIN,sector.get_point_viz()]
#         self.viz_pub.publish(self.best_sec_marker)
#
#
#     def publish_velocity(self,velocity:Vector):
#         pt = Point()
#         # velocity mighyt be r and theta
#         pt.x = velocity.x
#         pt.y = velocity.y
#         self.velocity_marker.points = [ORIGIN,pt]
#         self.viz_pub.publish(self.velocity_marker)


# def main(args):
#     rospy.init_node("VFH_node", anonymous=True)
#     vfh = VFH()
#     rospy.sleep(0.1)
#     rospy.spin()

# if __name__=='__main__':
# 	main(sys.argv)
