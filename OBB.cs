using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing.Drawing2D;
using YOLO.Extentions;

namespace YOLO
{
    public class OBB
    {
        InferenceSession InferenceSession { get; set; }
        string[] OutputData { get; set; }
        int Imgsz { get; set; }
        float Imgsz_inv { get; set; }
        readonly int MAX_POSSIBLE_OBJECT;
        readonly int N_CLASS;
        readonly int col_len;
        Dictionary<string, Color>? Labels { get; set; }
        Bitmap resized_img { get; set; }
        Graphics graphics { get; set; }
        NamedOnnxValue[] namedOnnxValues { get; set; }
        Dictionary<int, int> col_len_cache { get; set; }
        public OBB(string model_path, bool use_cuda)
        {
            if (use_cuda)
            {
                OrtCUDAProviderOptions cudaProviderOptions = new();
                cudaProviderOptions.UpdateOptions(new Dictionary<string, string>()
                {
                    { "cudnn_conv_use_max_workspace", "1" },
                    { "cudnn_conv1d_pad_to_nc1d", "1" },
                    { "arena_extend_strategy", "kSameAsRequested" },
                    { "do_copy_in_default_stream", "1" }
                });
                SessionOptions sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
                sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                InferenceSession = new(model_path, sessionOptions);
            }
            else
            {
                InferenceSession = new(model_path);
            }
            Imgsz = InferenceSession.InputMetadata["images"].Dimensions[2];
            Imgsz_inv = 1f / Imgsz;
            MAX_POSSIBLE_OBJECT = InferenceSession.OutputMetadata.ElementAt(0).Value.Dimensions[2];
            OutputData = InferenceSession.OutputMetadata.Keys.ToArray();
            col_len = InferenceSession.OutputMetadata.ElementAt(0).Value.Dimensions[1];
            N_CLASS = col_len - 5;
            resized_img = new(Imgsz, Imgsz);
            graphics = Graphics.FromImage(resized_img);
            graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
            namedOnnxValues = new NamedOnnxValue[1];
            using Bitmap bitmap = new(Imgsz, Imgsz);
            namedOnnxValues[0] = NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(bitmap));
            InferenceSession.Run(namedOnnxValues, OutputData);
            col_len_cache = new();
            for (int i = 0; i < 21; i++)
            {
                col_len_cache.Add(i, i * MAX_POSSIBLE_OBJECT);
            }
        }

        public List<OBBPrediction> Predict(Bitmap image, Dictionary<string, float> class_conf, float conf, float iou_conf)
        {
            float x_scaler = image.Width * Imgsz_inv;
            float y_scaler = image.Height * Imgsz_inv;
            List<OBBPrediction> predictions = new();
            ResizeImage(image);
            namedOnnxValues[0] = NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(resized_img));
            Tensor<float> output = InferenceSession.Run(namedOnnxValues, OutputData).ElementAt(0).AsTensor<float>();
            for (int j = 0; j < MAX_POSSIBLE_OBJECT; j++)
            {
                float max_score = 0f;
                int max_score_idx = 0;
                for (int i = 4; i < col_len - 1; i++)
                {
                    float value = output.ElementAt(col_len_cache[i] + j);
                    if (value > max_score)
                    {
                        max_score = value;
                        max_score_idx = i - 4;
                        if (max_score >= .5f)
                        {
                            break;
                        }
                    }
                }
                if (max_score > conf)
                {
                    YoloLabel label = new(max_score_idx, Labels.ElementAt(max_score_idx).Key, Labels.ElementAt(max_score_idx).Value);
                    RectangleF rectangle = new((output.ElementAt(col_len_cache[0] + j) - output.ElementAt(col_len_cache[2] + j) * .5f) * x_scaler,
                                               (output.ElementAt(col_len_cache[1] + j) - output.ElementAt(col_len_cache[3] + j) * .5f) * y_scaler,
                                               output.ElementAt(col_len_cache[2] + j) * x_scaler, output.ElementAt(col_len_cache[3] + j) * y_scaler);
                    float angle = output.ElementAt(col_len_cache[col_len - 1] + j);
                    OBBPrediction prediction = new(label, rectangle, angle, max_score);
                    predictions.Add(prediction);
                }
            }
            return Suppress(predictions, iou_conf);
        }

        private List<OBBPrediction> Suppress(List<OBBPrediction> items, float iou_conf)
        {
            List<OBBPrediction> result = new(items);
            foreach (OBBPrediction item in items)
            {
                foreach (OBBPrediction current in result.ToList()) // make a copy for each iteration
                {
                    if (current != item)
                    {
                        float intArea = RectangleF.Intersect(item.Rectangle, current.Rectangle).Area();
                        if ((intArea / (item.Area + current.Area - intArea)) >= iou_conf)
                        {
                            if (item.Score >= current.Score)
                            {
                                result.Remove(current);
                            }
                        }
                    }
                }
            }
            return result;
        }

        public void ResizeImage(Image image)
        {
            graphics.DrawImage(image, 0, 0, Imgsz, Imgsz);
        }

        public void SetupLabels(Dictionary<string, Color> labels)
        {
            Labels = labels;
        }

        public int GetModelNClass()
        {
            return N_CLASS;
        }
    }
}
