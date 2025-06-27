#include <wiringPi.h>
#include <softTone.h>
#include <cmath>
#include "functions.h"

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
void runMusicThread(std::atomic<bool>& running, std::atomic<bool>& sleeping) {
  constexpr int SPKR = 6;
  softToneCreate(SPKR);

  while (running) {
    if (sleeping) {
      softToneWrite(SPKR, 391);
      std::this_thread::sleep_for(std::chrono::milliseconds(280));
    }
    else {
      softToneWrite(SPKR, 0);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  softToneWrite(SPKR, 0);
}
void runLedBlinkingThread(std::atomic<bool>& running, std::atomic<bool>& sleeping, int gpioPin) {
  pinMode(gpioPin, OUTPUT);

  while (running) {
    if (sleeping) {
      digitalWrite(gpioPin, HIGH);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      digitalWrite(gpioPin, LOW);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    else {
      digitalWrite(gpioPin, LOW);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  digitalWrite(gpioPin, LOW); // turn off LED on exit
}
