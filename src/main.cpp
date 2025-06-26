#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "functions.h"

// ---------- 전역 상수 ---------- (컴파일 시에 값 결정)
constexpr double SCALE = 0.65; // 입력 프레임 축소 비율 (속도와 비례. 감지 성능과 반비례)
constexpr double INV_SCALE = 1.0 / SCALE; // 좌표 원복 시 사용
constexpr int    UPDATE_N = 3;     // N 프레임마다 얼굴 박스 새로 탐색
constexpr double EAR_THRESH = 0.18; // EAR 임계값 (높을수록 예민해짐)
// -------------------------------

int main() {
  cv::VideoCapture cap(0, cv::CAP_V4L2); // 카메라 열기
  if (!cap.isOpened()) { std::cerr << "카메라 열기 실패!\n"; return -1; } // 연결 실패 처리

  // ---------------- dlib 얼굴 검출 및 랜드마크 준비 ----------------
  dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
  dlib::shape_predictor sp;
  dlib::deserialize("../eye_data/shape_predictor_68_face_landmarks.dat") >> sp;

  cv::namedWindow("Window");
  // "Window" 라는 이름의 새로운 윈도우 생성.
  // 나중에 cv::imshow()로 출력하면 이 창에 영상이 표시된다.

  unsigned int frameCount = 0; // 총 처리 프레임 수
  bool hasFace = false;
  dlib::rectangle biggestFaceRect; // 감지된 얼굴 중 가장 큰 얼굴을 감싸는 사각형

  while (true) {
    auto t0 = std::chrono::high_resolution_clock::now(); // 프레임 타이머 시작

    cv::Mat frame; cap >> frame;
    if (frame.empty()) break;

    // 프레임 축소
    cv::Mat smallFrame;
    cv::resize(frame, smallFrame, cv::Size(), SCALE, SCALE, cv::INTER_AREA);

    // dlib API에서 사용 가능한 타입으로 변환
    dlib::cv_image<dlib::bgr_pixel> dlibSmallFrame(smallFrame);

    // N 프레임마다 얼굴 재탐색
    if (frameCount % UPDATE_N == 0) {
      std::vector<dlib::rectangle> faces = detector(dlibSmallFrame);
      if (faces.empty()) {
        hasFace = false;
      }
      else if (faces.size() == 1) {
        biggestFaceRect = faces[0];
        hasFace = true;
      }
      else {
        // 가장 면적 큰 얼굴 하나 찾기
        biggestFaceRect = *std::max_element(
          faces.begin(), faces.end(),
          [](const dlib::rectangle& a, const dlib::rectangle& b) {
            return a.area() < b.area();
          });
        hasFace = true;
      }
    }

    if (hasFace) {
      // 랜드마크 추출 (매프레임 수행: 박스는 최신/캐싱)
      dlib::full_object_detection shapeSmall = sp(dlibSmallFrame, biggestFaceRect);

      // 양쪽 눈 평균 EAR 계산 (36 ~ 41: 왼쪽 눈, 42 ~ 47: 오른쪽 눈)
      double earL = computeEAR(shapeSmall, 36);
      double earR = computeEAR(shapeSmall, 42);
      double earAvg = (earL + earR) / 2.0;

      // 시각화용 람다: 축소 좌표 -> 원본 좌표
      auto scaleVal = [](const long& x) {
        return static_cast<int>(x * INV_SCALE + 0.5);
        };

      auto scalePt = [&scaleVal](const dlib::point& p) {
        return cv::Point(scaleVal(p.x()), scaleVal(p.y()));
        };

      // 얼굴 박스 (녹색)
      cv::rectangle(frame,
        cv::Point(scaleVal(biggestFaceRect.left()), scaleVal(biggestFaceRect.top())),
        cv::Point(scaleVal(biggestFaceRect.right()), scaleVal(biggestFaceRect.bottom())),
        cv::Scalar(0, 255, 0), 2);

      // 눈 랜드마크 점 (파란색)
      for (int i = 36; i <= 47; ++i) {
        cv::circle(frame, scalePt(shapeSmall.part(i)), 2, cv::Scalar(255, 0, 0), -1);
      }

      // EAR 임계값 아래면 문구 출력 (빨간색)
      if (earAvg < EAR_THRESH) {
        cv::putText(frame, "Eye Closed!!!!",
          cv::Point(scaleVal(biggestFaceRect.left()), scaleVal(biggestFaceRect.top()) - 10),
          cv::FONT_HERSHEY_SIMPLEX, 0.9,
          cv::Scalar(0, 0, 255), 2);
      }
    }

    // FPS 출력
    double ms = std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - t0).count();
    double fps = (ms > 0.0) ? 1000.0 / ms : 0.0;
    std::ostringstream oss;
    oss << "FPS: " << std::fixed << std::setprecision(1) << fps;
    cv::putText(frame, oss.str(), cv::Point(10, 30),
      cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    cv::imshow("Window", frame);
    if (cv::waitKey(1) == 27) break;     // ESC 종료
    ++frameCount;
  }
  return 0;
}