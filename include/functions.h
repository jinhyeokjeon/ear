#pragma once

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

// EAR 값 계산
double computeEAR(const dlib::full_object_detection& s, int idx);
// 얼굴 검출용 백그라운드 스레드 함수 선언
void runFaceDetectionThread(std::atomic<bool>& running, cv::Mat& sharedFrame, std::mutex& frameMutex, dlib::rectangle& biggestFaceRect, bool& hasFace, std::mutex& faceMutex);
// LED 및 스피커 제어 함수
void runMusicThread(std::atomic<bool>& running, std::atomic<bool>& sleeping);
void runLedBlinkingThread(std::atomic<bool>& running, std::atomic<bool>& sleeping, int gpioPin);
