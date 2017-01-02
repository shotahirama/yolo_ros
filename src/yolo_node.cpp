/*
MIT License

Copyright (c) 2017 Shota Hirama

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "yolo_ros/DetectObject.h"
#include "yolo_ros/DetectObjectArray.h"
#include "yolo_ros/yolo.h"
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <image_transport/image_transport.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>

class YOLORos {
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  ros::Publisher objpub_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  Darknet darknet_;
  double thresh_;

  void imageCb(const sensor_msgs::ImageConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat cv_image = cv_ptr->image;
    yolo_ros::DetectObjectArray detectobjarray = darknet_.detect(cv_image);
    cv_ptr->image = darknet_.draw_rectbox(cv_image, detectobjarray);

    objpub_.publish(detectobjarray);
    image_pub_.publish(cv_ptr->toImageMsg());
  }

public:
  YOLORos() : nh_("~"), it_(nh_) {
    objpub_ = nh_.advertise<yolo_ros::DetectObjectArray>("detect_object", 1);
    image_sub_ = it_.subscribe("/camera/image_raw", 1, &YOLORos::imageCb, this);
    image_pub_ = it_.advertise("image_raw", 1);
    nh_.param("thresh", thresh_, 0.24);

    ROS_INFO("thresh : %f", thresh_);

    std::string package_path = ros::package::getPath("yolo_ros") + "/darknet";
    std::string cfgfile = package_path + "/cfg/yolo.cfg";
    std::string weightfile = package_path + "/yolo.weights";
    std::string namesfile = package_path + "/data/coco.names";
    darknet_.load(cfgfile, weightfile, namesfile);
    darknet_.set_thresh(thresh_);
  }
};

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "yolo_node");
  YOLORos dr;
  ros::spin();
  return 0;
}
