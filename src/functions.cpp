#include "functions.h"
#include <cmath>
#include <string>

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
void runFaceDetectionThread(std::atomic<bool>& running, dlib::frontal_face_detector& detector, cv::Mat& sharedFrame, std::mutex& frameMutex, dlib::rectangle& biggestFaceRect, bool& hasFace, std::mutex& faceMutex) {
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
void runFaceDetectionThread(std::atomic<bool>& running, dlib::frontal_face_detector& detector, cv::Mat& sharedFrame, std::mutex& frameMutex, dlib::rectangle& biggestFaceRect, bool& hasFace, std::mutex& faceMutex) {
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

extern std::vector<int> landmarkIdx;
double calibrateEARThreshold(cv::VideoCapture& cap, dlib::frontal_face_detector& detector, dlib::shape_predictor& sp) {
  // Open your eyes 단계
  while (true) {
    cv::Mat tempFrame;
    cap >> tempFrame;
    if (tempFrame.empty()) continue;
    cv::putText(tempFrame, "Open your eyes.", cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2); cv::imshow("Frame", tempFrame);
    cv::putText(tempFrame, "Press ENTER when you are ready", cv::Point(30, 100), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2); cv::imshow("Frame", tempFrame);

    dlib::cv_image<dlib::bgr_pixel> dlibFrame(tempFrame);
    std::vector<dlib::rectangle> faces = detector(dlibFrame);
    if (!faces.empty()) {
      dlib::full_object_detection landmarks = sp(dlibFrame, faces[0]);

      cv::rectangle(tempFrame,
        cv::Point(faces[0].left(), faces[0].top()),
        cv::Point(faces[0].right(), faces[0].bottom()),
        cv::Scalar(0, 255, 0), 2);

      for (int i = 0; i < landmarkIdx.size(); ++i) {
        dlib::point p = landmarks.part(landmarkIdx[i]);
        cv::circle(tempFrame, cv::Point(p.x(), p.y()), 2, cv::Scalar(255, 0, 0), -1);
      }
    }

    cv::imshow("Frame", tempFrame);
    if (cv::waitKey(30) == 13) break;
  }

  double openEarSum = 0.0;
  int openEarCount = 0;
  auto openStart = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - openStart < std::chrono::seconds(2)) {
    cv::Mat tempFrame;
    cap >> tempFrame;
    if (tempFrame.empty()) continue;

    dlib::cv_image<dlib::bgr_pixel> dlibFrame(tempFrame);
    std::vector<dlib::rectangle> faces = detector(dlibFrame);
    if (!faces.empty()) {
      dlib::full_object_detection landmarks = sp(dlibFrame, faces[0]);

      cv::rectangle(tempFrame,
        cv::Point(faces[0].left(), faces[0].top()),
        cv::Point(faces[0].right(), faces[0].bottom()),
        cv::Scalar(0, 255, 0), 2);

      for (int i = 0; i < landmarkIdx.size(); ++i) {
        dlib::point p = landmarks.part(landmarkIdx[i]);
        cv::circle(tempFrame, cv::Point(p.x(), p.y()), 2, cv::Scalar(255, 0, 0), -1);
      }

      double ear = (computeEAR(landmarks, 36) + computeEAR(landmarks, 42)) / 2.0;
      openEarSum += ear;
      openEarCount++;
    }
    double secondsElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - openStart).count() / 1000.0;
    std::string timerText = "Time: " + std::to_string(secondsElapsed).substr(0, 4) + "s";
    cv::putText(tempFrame, timerText, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    cv::imshow("Frame", tempFrame);
    cv::waitKey(1);
  }
  double openEAR = (openEarCount > 0) ? openEarSum / openEarCount : 0.3;

  // Close your eyes 단계
  while (true) {
    cv::Mat tempFrame;
    cap >> tempFrame;
    if (tempFrame.empty()) continue;
    cv::putText(tempFrame, "Close your eyes.", cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2); cv::imshow("Frame", tempFrame);
    cv::putText(tempFrame, "Press ENTER when you are ready", cv::Point(30, 100), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2); cv::imshow("Frame", tempFrame);
    dlib::cv_image<dlib::bgr_pixel> dlibFrame(tempFrame);
    std::vector<dlib::rectangle> faces = detector(dlibFrame);
    if (!faces.empty()) {
      dlib::full_object_detection landmarks = sp(dlibFrame, faces[0]);

      cv::rectangle(tempFrame,
        cv::Point(faces[0].left(), faces[0].top()),
        cv::Point(faces[0].right(), faces[0].bottom()),
        cv::Scalar(0, 255, 0), 2);

      for (int i = 0; i < landmarkIdx.size(); ++i) {
        dlib::point p = landmarks.part(landmarkIdx[i]);
        cv::circle(tempFrame, cv::Point(p.x(), p.y()), 2, cv::Scalar(255, 0, 0), -1);
      }
    }
    cv::imshow("Frame", tempFrame);
    if (cv::waitKey(30) == 13) break;
  }

  double closeEarSum = 0.0;
  int closeEarCount = 0;
  auto closeStart = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - closeStart < std::chrono::seconds(2)) {
    cv::Mat tempFrame;
    cap >> tempFrame;
    if (tempFrame.empty()) continue;

    dlib::cv_image<dlib::bgr_pixel> dlibFrame(tempFrame);
    std::vector<dlib::rectangle> faces = detector(dlibFrame);
    if (!faces.empty()) {
      dlib::full_object_detection landmarks = sp(dlibFrame, faces[0]);

      cv::rectangle(tempFrame,
        cv::Point(faces[0].left(), faces[0].top()),
        cv::Point(faces[0].right(), faces[0].bottom()),
        cv::Scalar(0, 255, 0), 2);

      for (int i = 0; i < landmarkIdx.size(); ++i) {
        dlib::point p = landmarks.part(landmarkIdx[i]);
        cv::circle(tempFrame, cv::Point(p.x(), p.y()), 2, cv::Scalar(255, 0, 0), -1);
      }

      double ear = (computeEAR(landmarks, 36) + computeEAR(landmarks, 42)) / 2.0;
      closeEarSum += ear;
      closeEarCount++;
    }
    double secondsElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - closeStart).count() / 1000.0;
    std::string timerText = "Time: " + std::to_string(secondsElapsed).substr(0, 4) + "s";
    cv::putText(tempFrame, timerText, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    cv::imshow("Frame", tempFrame);
    cv::waitKey(1);
  }
  double closeEAR = (closeEarCount > 0) ? closeEarSum / closeEarCount : 0.2;

  double EAR_THRESH = closeEAR + (openEAR - closeEAR) * 0.2;
  std::string str1 = "Open EAR: " + std::to_string(openEAR) + ", Close EAR: " + std::to_string(closeEAR);
  std::string str2 = "Threshold: " + std::to_string(EAR_THRESH);

  // 최종 시작 전 대기
  while (true) {
    cv::Mat tempFrame;
    cap >> tempFrame;
    if (tempFrame.empty()) continue;
    cv::putText(tempFrame, "Calibration done. Press ENTER to start app", cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    cv::putText(tempFrame, str1, cv::Point(30, 100), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    cv::putText(tempFrame, str2, cv::Point(30, 150), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    cv::imshow("Frame", tempFrame);
    if (cv::waitKey(30) == 13) break;
  }

  return closeEAR + (openEAR - closeEAR) * 0.2;
}