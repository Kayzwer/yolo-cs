# Usage

```
Yolov8 yolov8 = new("path/to/.onnx");
yolov8.SetupLabels(<dict(str, color)>)
yolov8.Predict(<Image>, <dict(str, color)>, <confidence_level>, <iou_level>)
```
Note: for the Yolov9.cs, it supports this [yolov9](https://github.com/WongKinYiu/yolov9), for the yolov9 from [ultralyutics](https://github.com/ultralytics/ultralytics), Yolov8.cs will work just fine as they are using the same output head.
