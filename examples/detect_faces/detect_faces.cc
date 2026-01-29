// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/timer.h"
#include "libs/camera/camera.h"
#include "libs/tensorflow/detection.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

// Runs face detection on the Edge TPU, using the on-board camera, printing
// results to the serial console and turning on the User LED when a face
// is detected.
//
// To build and flash from coralmicro root:
//    bash build.sh
//    python3 scripts/flashtool.py -e face_detection

namespace coralmicro {
namespace {
constexpr char kModelPath[] =
    // "/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite";
    "/models/multi_int8new_edgetpu.tflite";
constexpr int kTopK = 5;
constexpr float kThreshold = 0.5;

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

[[noreturn]] void Main() {

    vTaskDelay(pdMS_TO_TICKS(2000));  // wait 2s so serial is ready

  printf("Face Detection Example!\r\n");
  // Turn on Status LED to show the board is on.
  LedSet(Led::kStatus, true);

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }

  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    printf("ERROR: Failed to get EdgeTpu context\r\n");
    vTaskSuspend(nullptr);
  }

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddDequantize();
  resolver.AddDetectionPostprocess();
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena, kTensorArenaSize,
                                       &error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }

  if (interpreter.inputs().size() != 2) {
    printf("ERROR: Model must have only two input tensor and not %d \r\n",interpreter.inputs().size());
    vTaskSuspend(nullptr);
  }

  // Starting Camera.
  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);

  auto* input_tensor_image = interpreter.input_tensor(0);
  auto* input_tensor_question = interpreter.input_tensor(1);
  int model_img_height = input_tensor_image->dims->data[1];
  int model_img_width = input_tensor_image->dims->data[2];
  int model_q_height = input_tensor_question->dims->data[1];
  int model_q_width = input_tensor_question->dims->data[2];
  printf("[DEBUG]]: model image input height %d width %d \r\n",model_img_height,model_img_width);
  printf("[DEBUG]]: model question input height %d width %d \r\n",model_q_height,model_q_width);

  auto preprocess_start = TimerMillis();
  while (true) {
    CameraFrameFormat fmt{CameraFormat::kRgb,
                          CameraFilterMethod::kBilinear,
                          CameraRotation::k270,
                          model_img_width,
                          model_img_height,
                          false,
                          tflite::GetTensorData<uint8_t>(input_tensor_image)};
    if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
      printf("Failed to capture image\r\n");
      vTaskSuspend(nullptr);
    }

    if (interpreter.Invoke() != kTfLiteOk) {
      printf("Failed to invoke\r\n");
      vTaskSuspend(nullptr);
    }

    if (auto results =
            tensorflow::GetDetectionResults(&interpreter, kThreshold, kTopK);
        !results.empty()) {
      // printf("Found %d face(s):\r\n%s\r\n", results.size(),
            //  tensorflow::FormatDetectionOutput(results).c_str());
      LedSet(Led::kUser, true);
    } else {
      LedSet(Led::kUser, false);
    }
  }
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
