using System.Drawing;
using YOLO.Extentions;


namespace YOLO
{
    public class OBBPrediction
    {
        public YoloLabel Label { get; set; }
        public RectangleF Rectangle { get; set; }
        public float Area { get; set; }
        public float Score { get; set; }
        public float Angle { get; set; }

        public OBBPrediction(YoloLabel label, RectangleF rectangle, float angle, float confidence)
        {
            Label = label;
            Score = confidence;
            Rectangle = rectangle;
            Angle = angle;
            Area = rectangle.Area();
        }
    }
}
