#include "functions.h"
#include <cmath>

#define DNN 0

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

#if DNN
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
    int maxLen = 0;
    dlib::rectangle largest;

    for (int i = 0; i < detections.rows; ++i) {
      float confidence = detections.at<float>(i, 2);
      if (confidence > 0.4f) {
        int x1 = static_cast<int>(detections.at<float>(i, 3) * localFrame.cols);
        int y1 = static_cast<int>(detections.at<float>(i, 4) * localFrame.rows);
        int x2 = static_cast<int>(detections.at<float>(i, 5) * localFrame.cols);
        int y2 = static_cast<int>(detections.at<float>(i, 6) * localFrame.rows);

        int cx = (x1 + x2) >> 1;
        int cy = (y1 + y2) >> 1;
        int w = x2 - x1;
        int h = y2 - y1;
        int squareHalfLen = (std::max(w, h)) >> 1;

        // 정사각형 영역 계산
        int sq_x1 = std::max(0, cx - squareHalfLen);
        int sq_y1 = std::max(0, cy - squareHalfLen);
        int sq_x2 = std::min(localFrame.cols - 1, cx + squareHalfLen);
        int sq_y2 = std::min(localFrame.rows - 1, cy + squareHalfLen);

        int len = (sq_x2 - sq_x1);
        if (len > maxLen) {
          maxLen = len;
          largest = dlib::rectangle(sq_x1, sq_y1, sq_x2, sq_y2);
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(faceMutex);
      if (maxLen > 0) {
        biggestFaceRect = largest;
        hasFace = true;
      }
      else {
        hasFace = false;
      }
    }

  }
}
#else
void runFaceDetectionThread(std::atomic<bool>& running, cv::Mat& sharedFrame, std::mutex& frameMutex, dlib::rectangle& biggestFaceRect, bool& hasFace, std::mutex& faceMutex) {
  dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
  while (running) {
    cv::Mat localFrame;
    {
      std::lock_guard<std::mutex> lock(frameMutex);
      if (sharedFrame.empty()) continue;
      sharedFrame.copyTo(localFrame);
    }

    dlib::cv_image<dlib::bgr_pixel> dlibFrame(localFrame);
    std::vector<dlib::rectangle> faces = detector(dlibFrame);

    {
      std::lock_guard<std::mutex> lock(faceMutex);
      if (faces.empty()) {
        hasFace = false;
      }
      else if (faces.size() == 1) {
        biggestFaceRect = faces[0];
        hasFace = true;
      }
      else {
        biggestFaceRect = *std::max_element(faces.begin(), faces.end(),
          [](const dlib::rectangle& a, const dlib::rectangle& b) {
            return a.area() < b.area();
          });
        hasFace = true;
      }
    }

    // std::this_thread::sleep_for(std::chrono::milliseconds(UPDATE_MS));
  }
}
#endif