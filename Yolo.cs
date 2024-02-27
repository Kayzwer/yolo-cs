namespace YOLO
{
    public abstract class Yolo
    {
        public abstract void SetupLabels(Dictionary<string, Color> color_mapper);

        public abstract List<YoloPrediction> Predict(Bitmap clone, Dictionary<string, float> class_conf, float conf_thres = 0, float iou_thres = 0);

        public abstract int GetModelNClass();
    }
}
