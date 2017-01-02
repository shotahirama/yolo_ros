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

#ifndef YOLO_H
#define YOLO_H

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "yolo_ros/DetectObject.h"
#include "yolo_ros/DetectObjectArray.h"

extern "C" {
#undef __cplusplus
#include "box.h"
#include "cost_layer.h"
#include "detection_layer.h"
#include "image.h"
#include "network.h"
#include "parser.h"
#include "region_layer.h"
#include "utils.h"
#define __cplusplus
}

class Darknet {
  network net_;
  float nms_;
  float thresh_;
  std::vector<box> boxes_;
  std::vector<float*> probs_;
  std::vector<std::string> names_;
  std::vector<cv::Scalar> rngclr_;

  image mat_to_image(cv::Mat src) {
    int h = src.rows;
    int w = src.cols;
    int c = src.channels();
    image out = make_image(w, h, c);
    int countdata = 0;
    for (int i = 0; i < c; i++) {
      for (int j = 0; j < h; j++) {
        for (int k = 0; k < w; k++) {
          out.data[countdata++] = src.data[j * src.step + k * c + i] / 255.0;
        }
      }
    }
    rgbgr_image(out);
    return out;
  }

 public:
  Darknet() : nms_(0.4), thresh_(0.24) {}

  void load(std::string cfgfile, std::string weightfile, std::string namesfile) {
    srand(2222222);
    std::string reading_buffer;
    std::ifstream ifs(namesfile.c_str());
    while (std::getline(ifs, reading_buffer)) {
      names_.push_back(reading_buffer);
      rngclr_.push_back(cv::Scalar(rand() % 255, rand() % 255, rand() % 255));
    }
    net_ = parse_network_cfg(const_cast<char*>(cfgfile.c_str()));
    load_weights(&net_, const_cast<char*>(weightfile.c_str()));
    set_batch_network(&net_, 1);
  }

  void set_thresh(float thresh) { this->thresh_ = thresh; }

  void set_nms(float nms) { this->nms_ = nms; }

  yolo_ros::DetectObjectArray detect(cv::Mat src) {
    image im = mat_to_image(src);
    image sized = resize_image(im, net_.w, net_.h);
    layer l = net_.layers[net_.n - 1];
    int netsize = l.w * l.h * l.n;
    boxes_.resize(netsize);
    probs_.resize(netsize);
    for (int i = 0; i < netsize; i++) {
      probs_[i] = static_cast<float*>(calloc(l.classes, sizeof(float)));
    }
    network_predict(net_, sized.data);
    if (l.type == DETECTION) {
      get_detection_boxes(l, 1, 1, thresh_, probs_.data(), boxes_.data(), 0);
    } else if (l.type == REGION) {
      get_region_boxes(l, 1, 1, thresh_, probs_.data(), boxes_.data(), 0, 0);
    }
    if (nms_) {
      do_nms(boxes_.data(), probs_.data(), netsize, l.classes, nms_);
    }
    yolo_ros::DetectObjectArray detectobjectarray;
    for (int i = 0; i < netsize; i++) {
      int class_id = max_index(probs_[i], l.classes);
      float prob = probs_[i][class_id];
      if (prob > thresh_) {
        yolo_ros::DetectObject detectobj;
        detectobj.name = names_[class_id];
        detectobj.id = class_id;
        detectobj.prob = prob;
        box b = boxes_[i];
        int left = (b.x - b.w / 2.) * im.w;
        int right = (b.x + b.w / 2.) * im.w;
        int top = (b.y - b.h / 2.) * im.h;
        int bot = (b.y + b.h / 2.) * im.h;

        if (left < 0) left = 0;
        if (right > im.w - 1) right = im.w - 1;
        if (top < 0) top = 0;
        if (bot > im.h - 1) bot = im.h - 1;
        detectobj.left = left;
        detectobj.top = top;
        detectobj.right = right;
        detectobj.bottom = bot;
        detectobjectarray.detectobject.push_back(detectobj);
      }
    }
    free_image(im);
    free_image(sized);
    for (int i = 0; i < probs_.size(); i++) {
      free(probs_[i]);
    }
    return detectobjectarray;
  }

  cv::Mat draw_rectbox(cv::Mat src, yolo_ros::DetectObjectArray detectobjarray) {
    for (yolo_ros::DetectObject detectobj : detectobjarray.detectobject) {
      cv::rectangle(src, cv::Point(detectobj.left, detectobj.top), cv::Point(detectobj.right, detectobj.bottom), rngclr_[detectobj.id], 4);
      int baseline = 0;
      cv::Size textsize = cv::getTextSize(detectobj.name, cv::FONT_HERSHEY_SIMPLEX, 0.8, 1, &baseline);
      cv::rectangle(src, cv::Point(detectobj.left - 2, detectobj.top), cv::Point(detectobj.left + textsize.width, detectobj.top - textsize.height - 15), rngclr_[detectobj.id], -1);
      cv::putText(src, detectobj.name, cv::Point(detectobj.left, detectobj.top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 1, CV_AA);
    }
    return src;
  }
};

#endif
