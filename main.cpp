#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <cmath>
#include <chrono>      // ⏱️ 프레임 시간 측정
#include <sstream>     // FPS 문자열
#include <iomanip>     // 소수점 포맷

// EAR 계산 함수 --------------------------------------------------------------
double computeEAR(const dlib::full_object_detection& shape, int startIdx) {
  auto p1 = shape.part(startIdx);
  auto p2 = shape.part(startIdx + 1);
  auto p3 = shape.part(startIdx + 2);
  auto p4 = shape.part(startIdx + 3);
  auto p5 = shape.part(startIdx + 4);
  auto p6 = shape.part(startIdx + 5);

  auto dist = [](const dlib::point& a, const dlib::point& b) {
    double dx = static_cast<double>(a.x() - b.x());
    double dy = static_cast<double>(a.y() - b.y());
    return std::sqrt(dx * dx + dy * dy);
    };

  double A = dist(p2, p6);
  double B = dist(p3, p5);
  double C = dist(p1, p4);
  return (A + B) / (2.0 * C);
}
// ---------------------------------------------------------------------------

int main() {
  cv::VideoCapture cap(0, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    std::cerr << "카메라 열기 실패!" << std::endl;
    return -1;
  }

  dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
  dlib::shape_predictor sp;
  dlib::deserialize("../eye_data/shape_predictor_68_face_landmarks.dat") >> sp;

  const double EAR_THRESH = 0.18;  // 눈 감김 임계값(필요 시 조정)
  cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

  cv::Mat frame;
  while (true) {
    auto t0 = std::chrono::high_resolution_clock::now();  // ⏱️ 시작

    cap >> frame;
    if (frame.empty()) break;

    dlib::cv_image<dlib::bgr_pixel> cimg(frame);
    std::vector<dlib::rectangle> faces = detector(cimg);

    for (const auto& face : faces) {
      // 랜드마크 추출
      dlib::full_object_detection shape = sp(cimg, face);

      // EAR 계산
      double earLeft = computeEAR(shape, 36);
      double earRight = computeEAR(shape, 42);
      double earAvg = (earLeft + earRight) / 2.0;

      // 눈 랜드마크 시각화
      for (int i = 36; i <= 47; ++i) {
        cv::circle(frame,
          cv::Point(shape.part(i).x(), shape.part(i).y()),
          2, cv::Scalar(255, 0, 0), -1);
      }

      // 얼굴 박스
      cv::rectangle(frame,
        cv::Point(face.left(), face.top()),
        cv::Point(face.right(), face.bottom()),
        cv::Scalar(0, 255, 0), 2);

      // 눈 감김 판단
      if (earAvg < EAR_THRESH) {
        cv::putText(frame, "Eye Closed!!!!",
          cv::Point(face.left(), face.top() - 10),
          cv::FONT_HERSHEY_SIMPLEX, 0.9,
          cv::Scalar(0, 0, 255), 2);
      }
    }

    // ⏱️ 끝 + FPS 계산
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double fps = (ms > 0.0) ? 1000.0 / ms : 0.0;

    // FPS 텍스트 렌더링
    std::ostringstream oss;
    oss << "FPS: " << std::fixed << std::setprecision(1) << fps;
    cv::putText(frame, oss.str(), cv::Point(10, 30),
      cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(255, 255, 255), 2);

    // 화면 출력
    cv::imshow("Camera", frame);
    if (cv::waitKey(1) == 27) break;  // ESC 키 종료
  }
  return 0;
}