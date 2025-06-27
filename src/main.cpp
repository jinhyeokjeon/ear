#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>
#include "functions.h"

// ---------- 전역 상수 ----------
constexpr int    UPDATE_MS = 100;     // 얼굴 검출 주기 (ms 단위)
constexpr double EAR_THRESH = 0.20; // EAR 임계값
// -------------------------------

// 얼굴 검출용 백그라운드 스레드 함수 선언
void runFaceDetectionThread(
  std::atomic<bool>& running,
  cv::Mat& sharedFrame,
  std::mutex& frameMutex,
  dlib::rectangle& biggestFaceRect,
  bool& hasFace,
  std::mutex& faceMutex
);

// ---------------------------- main ----------------------------
int main() {
  cv::VideoCapture cap(0, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    std::cerr << "카메라 열기 실패!\n";
    return -1;
  }

  // dlib 초기화
  dlib::shape_predictor sp;
  dlib::deserialize("../eye_data/shape_predictor_68_face_landmarks.dat") >> sp;

  // 공유 변수
  dlib::rectangle biggestFaceRect;
  bool hasFace = false;
  std::mutex faceMutex;

  cv::Mat sharedFrame;
  std::mutex frameMutex;

  std::atomic<bool> running = true;

  // 백그라운드 얼굴 탐지 스레드 시작
  std::thread faceThread(runFaceDetectionThread,
    std::ref(running),
    std::ref(sharedFrame),
    std::ref(frameMutex),
    std::ref(biggestFaceRect),
    std::ref(hasFace),
    std::ref(faceMutex)
  );

  cv::namedWindow("Window");
  unsigned int frameCount = 0;

  while (true) {
    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) break;

    // 최신 프레임 공유
    {
      std::lock_guard<std::mutex> lock(frameMutex);
      frame.copyTo(sharedFrame);
    }

    // 얼굴 박스 가져오기
    dlib::rectangle faceRect;
    bool localHasFace;
    {
      std::lock_guard<std::mutex> lock(faceMutex);
      localHasFace = hasFace;
      faceRect = biggestFaceRect;
    }

    if (localHasFace) {
      dlib::cv_image<dlib::bgr_pixel> dlibFrame(frame);
      dlib::full_object_detection landmarks = sp(dlibFrame, faceRect);
      double earL = computeEAR(landmarks, 36);
      double earR = computeEAR(landmarks, 42);
      double earAvg = (earL + earR) / 2.0;

      // 얼굴 박스
      cv::rectangle(frame,
        cv::Point(faceRect.left(), faceRect.top()),
        cv::Point(faceRect.right(), faceRect.bottom()),
        cv::Scalar(0, 255, 0), 2);

      // 눈 랜드마크
      for (int i = 36; i <= 47; ++i) {
        dlib::point p = landmarks.part(i);
        cv::circle(frame, cv::Point(p.x(), p.y()), 2, cv::Scalar(255, 0, 0), -1);
      }

      if (earAvg < EAR_THRESH) {
        cv::putText(frame, "Eye Closed!!!!",
          cv::Point(faceRect.left(), faceRect.top() - 10),
          cv::FONT_HERSHEY_SIMPLEX, 0.9,
          cv::Scalar(0, 0, 255), 2);
      }
    }

    // FPS 계산
    double ms = std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - t0).count();
    double fps = (ms > 0.0) ? 1000.0 / ms : 0.0;
    std::ostringstream oss;
    oss << "FPS: " << std::fixed << std::setprecision(1) << fps;
    cv::putText(frame, oss.str(), cv::Point(10, 30),
      cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    cv::imshow("Window", frame);
    if (cv::waitKey(1) == 27) break;
    ++frameCount;
  }

  // 종료 처리
  running = false;
  faceThread.join();
  return 0;
}

void runFaceDetectionThread(
  std::atomic<bool>& running,
  cv::Mat& sharedFrame,
  std::mutex& frameMutex,
  dlib::rectangle& biggestFaceRect,
  bool& hasFace,
  std::mutex& faceMutex
) {
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