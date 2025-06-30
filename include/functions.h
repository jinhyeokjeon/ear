#pragma once

#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <iostream>
#include <vector>

// EAR 값 계산
double computeEAR(const dlib::full_object_detection& s, int idx);
// 얼굴 검출용 백그라운드 스레드 함수 선언
void runFaceDetectionThread(std::atomic<bool>& running, dlib::frontal_face_detector& detector, cv::Mat& sharedFrame, std::mutex& frameMutex, dlib::rectangle& biggestFaceRect, bool& hasFace, std::mutex& faceMutex);
double calibrateEARThreshold(cv::VideoCapture& cap, dlib::frontal_face_detector& detector, dlib::shape_predictor& sp);