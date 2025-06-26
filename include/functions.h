#pragma once

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

double computeEAR(const dlib::full_object_detection& s, int idx); // EAR 값 계산