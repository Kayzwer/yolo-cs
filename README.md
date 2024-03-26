# Usage
## YOLOv8 object detection

```
Yolov8 yolov8 = new("path/to/.onnx", false);
Color[] colors = []; // fill in the colors for classes.
yolov8.SetupColors(colors);
Image image = Image.FromFile("path/to/img");
List<YoloPrediction> predictions = yolov8.Preidct((Bitmap)image, .5f, .5f);
Utils.DrawBoundingBox(image, predictions, 2, 16); // this return a image with bouding boxes drawn.
```

## YOLOv8 oriented bounding box
```
OBB obb = new("path/to/.onnx", false);
Color[] colors = [];
obb.SetupColors(colors);
Image image = Image.FromFile("path/to/img");
List<OBBPrediction> predictions = obb.Predict((Bitmap)image, .5f, .5f);
Utils.DrawRotatedBoundingBox(image, predictions, 2, 16);
```

## RT-DETR object detection
```
RTDETR rtdetr = new("path/to/.onnx", false);
Color[] colors = [];
rtdetr.SetupColors(colors);
Image image = Image.FromFile("path/to/img");
List<OBBPrediction> predictions = rtdetr.Predict((Bitmap)image, .5f, .5f);
Utils.DrawBoundingBox(image, predictions, 2, 16);
```

## YOLOv9 object detection (https://github.com/WongKinYiu/yolov9)
```
Yolov9 yolov9 = new("path/to/.onnx", false);
Color[] colors = [];
yolov9.SetupColors(colors);
Image image = Image.FromFile("path/to/img");
List<YoloPrediction> predictions = yolov9.Preidct((Bitmap)image, .5f, .5f);
Utils.DrawBoundingBox(image, predictions, 2, 16);
```

Note: for the Yolov9.cs, it supports this [yolov9](https://github.com/WongKinYiu/yolov9), for the yolov9 from [ultralyutics](https://github.com/ultralytics/ultralytics), Yolov8.cs will work just fine as they are using the same output head.
