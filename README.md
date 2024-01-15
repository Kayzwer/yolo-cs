# Usage
Yolo
```
Yolov8 yolov8 = new("path/to/.onnx");
yolov8.SetupLabels(<dict(str, color)>);
List<YoloPrediction> predictions = yolov8.Predict(<Image>, <dict(str, float)>, <confidence_level>, <iou_level>);
```
RTDETR
```
RTDETR rtdetr = new("path/to/.onnx");
rtdetr.SetupLabels(<dict(str, color)>);
List<YoloPrediction> predictions = rtdetr.Predict(<Image>, <dict(str, float)>, <confidence_level>, <iou_level>);
```
OBB
```
OBB obb = new("path/to/.onnx");
obb.SetupLabels(<dict(str, color)>);
List<OBBPrediction> predictions = obb.Predict(<Image>, <dict(str, float)>, <confidence_level>, <iou_level>);
```
`<dict(str, float)>` is to control confidence level for each of the classes,
`<confidence_level>` is to control confidence level for all classes
