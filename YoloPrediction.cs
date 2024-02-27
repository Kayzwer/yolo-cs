using YOLO.Extentions;
using System.Drawing;


namespace YOLO
{
    public class YoloPrediction
    {
        public YoloLabel Label { get; set; }
        public RectangleF Rectangle { get; set; }
        public float Area { get; set; }
        public float Score { get; set; }

        public YoloPrediction(YoloLabel label, RectangleF rectangle, float confidence)
        {
            Label = label;
            Score = confidence;
            Rectangle = rectangle;
            Area = rectangle.Area();
        }
    }
}
