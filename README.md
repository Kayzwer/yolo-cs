# Usage

```
Yolov8 yolov8 = new("path/to/.onnx");
yolov8.SetupLabels(<dict(str, color)>)
yolov8.Predict(<Image>, <dict(str, color)>, <confidence_level>, <iou_level>)
```
