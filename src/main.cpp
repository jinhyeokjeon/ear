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
#include <deque>
#include <utility>
#include <wiringPi.h>
#include "functions.h"

// ------------------------- 전역 상수 -------------------------
constexpr int    UPDATE_MS = 100;     // 얼굴 검출 주기 (ms 단위)
constexpr double EAR_THRESH = 0.30; // EAR 임계값
constexpr double BLINK_RATIO_THRESH = 0.6; // 감은 비율 임계값
constexpr int BLINK_WINDOW_MS = 2000; // 분석 시간 윈도우 (2초)
// -------------------------------------------------------------

int main() {
  // 카메라 열기
  cv::VideoCapture cap(0, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    std::cerr << "카메라 열기 실패!\n";
    return -1;
  }

  // wiringPi 초기화
  wiringPiSetup();

  // dlib 모델 불러오기
  dlib::shape_predictor sp;
  dlib::deserialize("../eye_data/shape_predictor_68_face_landmarks.dat") >> sp;

  // "Window" 창 생성. (이후 cv::imshow("Window", frame) 하면 여기에 출력된다)
  cv::namedWindow("Frame");

  // 쓰레드와의 공유 변수
  dlib::rectangle biggestFaceRect; // 가장 큰 얼굴 사각형
  bool hasFace = false; // 얼굴 감지 여부
  std::mutex faceMutex; // 위 두 변수 동기화 뮤텍스
  cv::Mat sharedFrame; // 쓰레드에게 전달할 프레임
  std::mutex frameMutex; // 위 변수 동기화 뮤텍스
  std::atomic<bool> sleeping = false; // 졸음 여부 저장 변수
  std::atomic<bool> running = true; // 프로그램 동작 여부 변수 (쓰레드 제어용)

  // 눈 감음 정보 저장 변수
  std::deque<std::pair<std::chrono::steady_clock::time_point, bool>> blinkHistory;
  unsigned long long closedCount = 0;

  // 얼굴 탐지 스레드 시작
  std::thread faceThread(runFaceDetectionThread, std::ref(running), std::ref(sharedFrame), std::ref(frameMutex), std::ref(biggestFaceRect), std::ref(hasFace), std::ref(faceMutex));

  // LED, 부저 울림 스레드 시작
  std::thread ledThread(runLedBlinkingThread, std::ref(running), std::ref(sleeping), 1);
  std::thread musicThread(runMusicThread, std::ref(running), std::ref(sleeping));

  while (true) {
    // FPS 계산 위한 시간값 저장
    auto t0 = std::chrono::high_resolution_clock::now();

    // 카메라로부터 한 프레임 받아옴
    cv::Mat frame; cap >> frame;
    if (frame.empty()) break;

    // 얼굴 영역 계산 쓰레드와 최신 프레임 공유
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

    // 얼굴 감지 성공시
    if (localHasFace) {
      // 68개의 얼굴 랜드마크 추출
      dlib::cv_image<dlib::bgr_pixel> dlibFrame(frame);
      dlib::full_object_detection landmarks = sp(dlibFrame, faceRect);

      // 얼굴 박스 표시
      cv::rectangle(frame,
        cv::Point(faceRect.left(), faceRect.top()),
        cv::Point(faceRect.right(), faceRect.bottom()),
        cv::Scalar(0, 255, 0), 2);

      // 눈 랜드마크 표시
      for (int i = 36; i <= 47; ++i) {
        dlib::point p = landmarks.part(i);
        cv::circle(frame, cv::Point(p.x(), p.y()), 2, cv::Scalar(255, 0, 0), -1);
      }

      // 랜드마크 이용해서 EAR 계산
      double earL = computeEAR(landmarks, 36);
      double earR = computeEAR(landmarks, 42);
      double earAvg = (earL + earR) / 2.0;

      // 눈 감음 여부 확인
      bool isClosed = (earAvg < EAR_THRESH);
      if (isClosed) ++closedCount;

      // { 현재 시간, 눈 감음 여부} 기록
      auto now = std::chrono::steady_clock::now();
      blinkHistory.emplace_back(now, isClosed);

      // 눈 감았을 시 경고 문구 출력
      if (isClosed) {
        cv::putText(frame, "Eye Closed!!!!",
          cv::Point(faceRect.left(), faceRect.top() - 10),
          cv::FONT_HERSHEY_SIMPLEX, 0.9,
          cv::Scalar(0, 0, 255), 2);
      }

      // 2초간의 { 현재 시간, 눈 감음 여부 } Window 에서 눈 감김 비율 계산 && 화면에 출력
      double ratio = static_cast<double>(closedCount) / blinkHistory.size();
      std::ostringstream ratioOss;
      ratioOss << "Eye Closed Ratio: " << std::fixed << std::setprecision(2) << ratio;
      cv::putText(frame, ratioOss.str(), cv::Point(10, 60),
        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

      // 눈 감김 비율이 임계치 넘을 시 경고
      if (ratio >= BLINK_RATIO_THRESH) {
        cv::putText(frame, "SLEEPING ! ! !", cv::Point(faceRect.left(), faceRect.top() - 40), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
        sleeping = true;
      }
      else {
        sleeping = false;
      }

      // 2초간의 윈도우 초과한 항목 제거
      while (!blinkHistory.empty() && std::chrono::duration_cast<std::chrono::milliseconds>(now - blinkHistory.front().first).count() > BLINK_WINDOW_MS) {
        if (blinkHistory.front().second) --closedCount;
        blinkHistory.pop_front();
      }
    }

    // FPS 계산
    double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    double fps = (ms > 0.0) ? 1000.0 / ms : 0.0;
    std::ostringstream oss;
    oss << "FPS: " << std::fixed << std::setprecision(1) << fps;
    cv::putText(frame, oss.str(), cv::Point(10, 30),
      cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    cv::imshow("Frame", frame);
    if (cv::waitKey(1) == 27) break;
  }

  // 종료 처리
  running = false;
  faceThread.join();
  return 0;
}