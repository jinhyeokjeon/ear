#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <cmath>
#include <chrono>
#include <sstream>
#include <iomanip>

constexpr double SCALE = 0.65;           // ⬇️ Dlib 계산용 축소 비율 (0.5 = 절반)
constexpr double INV_SCALE = 1.0 / SCALE;
constexpr double EAR_THRESH = 0.18;      // 눈 감김 임계값

// ──────────────────────────────────────────────────────────────
// EAR 계산 (축소된 좌표 그대로 넣어도 비율이니 상관없음)
double computeEAR(const dlib::full_object_detection& shape, int startIdx) {
  auto dist = [](const dlib::point& a, const dlib::point& b) {
    double dx = static_cast<double>(a.x() - b.x());
    double dy = static_cast<double>(a.y() - b.y());
    return std::sqrt(dx * dx + dy * dy);
    };
  double A = dist(shape.part(startIdx + 1), shape.part(startIdx + 5));
  double B = dist(shape.part(startIdx + 2), shape.part(startIdx + 4));
  double C = dist(shape.part(startIdx), shape.part(startIdx + 3));
  return (A + B) / (2.0 * C);
}
// ──────────────────────────────────────────────────────────────

int main() {
  cv::VideoCapture cap(0, cv::CAP_V4L2);
  if (!cap.isOpened()) { std::cerr << "카메라 열기 실패!\n"; return -1; }

  dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
  dlib::shape_predictor sp;
  dlib::deserialize("../eye_data/shape_predictor_68_face_landmarks.dat") >> sp;

  cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);
  cv::Mat frame, smallFrame;

  while (true) {
    auto t0 = std::chrono::high_resolution_clock::now();          // ⏱️ 시작

    cap >> frame;
    if (frame.empty()) break;

    // 1️⃣ 다운샘플
    cv::resize(frame, smallFrame, cv::Size(), SCALE, SCALE, cv::INTER_LINEAR);

    // 2️⃣ Dlib 처리 (저해상도)
    dlib::cv_image<dlib::bgr_pixel> cimgSmall(smallFrame);
    std::vector<dlib::rectangle> facesSmall = detector(cimgSmall);

    for (const auto& faceSmall : facesSmall) {
      // 랜드마크 추출(저해상도)
      dlib::full_object_detection shapeSmall = sp(cimgSmall, faceSmall);

      // EAR (스케일 영향 없음)
      double earLeft = computeEAR(shapeSmall, 36);
      double earRight = computeEAR(shapeSmall, 42);
      double earAvg = (earLeft + earRight) / 2.0;

      // 좌표 원본 해상도로 복원 후 시각화
      // 얼굴 박스
      cv::rectangle(frame,
        cv::Point(static_cast<int>(faceSmall.left() * INV_SCALE),
          static_cast<int>(faceSmall.top() * INV_SCALE)),
        cv::Point(static_cast<int>(faceSmall.right() * INV_SCALE),
          static_cast<int>(faceSmall.bottom() * INV_SCALE)),
        cv::Scalar(0, 255, 0), 2);

      // 랜드마크 36~47번(양쪽 눈) 표시
      for (int i = 36; i <= 47; ++i) {
        int x = static_cast<int>(shapeSmall.part(i).x() * INV_SCALE);
        int y = static_cast<int>(shapeSmall.part(i).y() * INV_SCALE);
        cv::circle(frame, cv::Point(x, y), 2, cv::Scalar(255, 0, 0), -1);
      }

      // 눈 감김 판단
      if (earAvg < EAR_THRESH) {
        int xText = static_cast<int>(faceSmall.left() * INV_SCALE);
        int yText = static_cast<int>(faceSmall.top() * INV_SCALE) - 10;
        cv::putText(frame, "Eye Closed!!!!",
          cv::Point(xText, yText),
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
      cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(255, 255, 255), 2);

    // 출력
    cv::imshow("Camera", frame);
    if (cv::waitKey(1) == 27) break;  // ESC
  }
  return 0;
}