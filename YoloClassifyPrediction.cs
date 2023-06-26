namespace YOLO
{
    public class YoloClassifyPrediction
    {
        public YoloLabel? Label { get; set; }
        public float Score { get; set; }

        public YoloClassifyPrediction(YoloLabel? label, float score)
        {
            Label = label;
            Score = score;
        }
    }
}
