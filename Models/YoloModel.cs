namespace YOLO.Models
{
    public class YoloModel
    {
        public int Dimensions { get; set; } //yolov7 包含nms 的模型不需要此参数

        public string[] Outputs { get; set; }

        public List<YoloLabel> Labels { get; set; } = new List<YoloLabel>();
    }
}
