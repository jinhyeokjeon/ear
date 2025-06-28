#include "functions.h"
#include <cmath>

double computeEAR(const dlib::full_object_detection& s, int idx) {
  auto dist = [](const dlib::point& a, const dlib::point& b) {
    double dx = a.x() - b.x(), dy = a.y() - b.y();
    return std::sqrt(dx * dx + dy * dy);
    };
  double A = dist(s.part(idx + 1), s.part(idx + 5));
  double B = dist(s.part(idx + 2), s.part(idx + 4));
  double C = dist(s.part(idx), s.part(idx + 3));
  return (A + B) / (2.0 * C);
}
void runFaceDetectionThread(std::atomic<bool>& running, cv::Mat& sharedFrame, std::mutex& frameMutex, dlib::rectangle& biggestFaceRect, bool& hasFace, std::mutex& faceMutex) {
  // DNN 기반 얼굴 검출기 로드
  cv::dnn::Net net = cv::dnn::readNetFromCaffe("../model/deploy.prototxt.txt", "../model/res10_300x300_ssd_iter_140000.caffemodel");

  while (running) {
    cv::Mat localFrame;
    {
      std::lock_guard<std::mutex> lock(frameMutex);
      if (sharedFrame.empty()) continue;
      sharedFrame.copyTo(localFrame);
    }

    // 프레임을 blob으로 변환
    cv::Mat blob = cv::dnn::blobFromImage(localFrame, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123));
    net.setInput(blob);
    cv::Mat output = net.forward();

    // 검출 결과 해석
    cv::Mat detections(output.size[2], output.size[3], CV_32F, output.ptr<float>());
    float maxArea = 0.0f;
    dlib::rectangle largest;

    for (int i = 0; i < detections.rows; ++i) {
      float confidence = detections.at<float>(i, 2);
      if (confidence > 0.4f) {
        int x1 = static_cast<int>(detections.at<float>(i, 3) * localFrame.cols);
        int y1 = static_cast<int>(detections.at<float>(i, 4) * localFrame.rows);
        int x2 = static_cast<int>(detections.at<float>(i, 5) * localFrame.cols);
        int y2 = static_cast<int>(detections.at<float>(i, 6) * localFrame.rows);

        float area = static_cast<float>((x2 - x1) * (y2 - y1));
        if (area > maxArea) {
          maxArea = area;
          largest = dlib::rectangle(x1, y1, x2, y2);
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(faceMutex);
      if (maxArea > 0.0f) {
        biggestFaceRect = largest;
        hasFace = true;
      }
      else {
        hasFace = false;
      }
    }

  }
}
