using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing.Drawing2D;
using YOLO.Extentions;
using System.Drawing;
using Newtonsoft.Json;


namespace YOLO
{
    public class Yolov9
    {
        InferenceSession InferenceSession { get; set; }
        string[] OutputData { get; set; }
        int Imgsz { get; set; }
        float Imgsz_inv { get; set; }
        readonly int MAX_POSSIBLE_OBJECT;
        readonly int col_len;
        Dictionary<string, Color>? Labels { get; set; }
        Bitmap resized_img { get; set; }
        Graphics graphics { get; set; }
        NamedOnnxValue[] namedOnnxValues { get; set; }
        Dictionary<int, int> col_len_caches { get; set; }
        public Yolov9(string model_path, bool use_cuda)
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
            Imgsz_inv = 1.0f / Imgsz;
            MAX_POSSIBLE_OBJECT = InferenceSession.OutputMetadata.ElementAt(0).Value.Dimensions[2];
            OutputData = InferenceSession.OutputMetadata.Keys.ToArray();
            col_len = InferenceSession.OutputMetadata.ElementAt(0).Value.Dimensions[1];
            resized_img = new(Imgsz, Imgsz);
            graphics = Graphics.FromImage(resized_img);
            graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
            namedOnnxValues = new NamedOnnxValue[1];
            using Bitmap bitmap = new(Imgsz, Imgsz);
            namedOnnxValues[0] = NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(bitmap));
            InferenceSession.Run(namedOnnxValues, OutputData);
            col_len_caches = [];
            Labels = [];
            for (int i = 0; i < col_len; i++)
            {
                col_len_caches.Add(i, i * MAX_POSSIBLE_OBJECT);
            }
        }

        public void SetupColors(Color[] colors)
        {
            Dictionary<int, string> classes = JsonConvert.DeserializeObject<Dictionary<int, string>>(InferenceSession.ModelMetadata.CustomMetadataMap["names"])!;
            for (int i = 0; i < colors.Length; i++)
            {
                Labels.Add(classes.ElementAt(i).Value, colors[i]);
            }
        }

        public List<YoloPrediction> Predict(Bitmap image, float conf, float iou_conf)
        {
            ResizeImage(image);
            namedOnnxValues[0] = NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(resized_img));
            return Suppress(GetBboxes_n_Scores(InferenceSession.Run(namedOnnxValues, OutputData).ElementAt(0).AsTensor<float>(),
                conf, image.Width, image.Height), iou_conf);
        }

        public List<YoloPrediction> GetBboxes_n_Scores(Tensor<float> input, float conf, int image_width, int image_height)
        {
            List<YoloPrediction> predictions = [];
            float width_scale = image_width * Imgsz_inv;
            float height_scale = image_height * Imgsz_inv;
            Parallel.For(0, MAX_POSSIBLE_OBJECT, i =>
            {
                float max_score = .0f;
                int max_score_idx = 0;
                for (int j = 4; j < col_len; j++)
                {
                    float value = input.ElementAt(col_len_caches[j] + i);
                    if (value > max_score)
                    {
                        max_score = value;
                        max_score_idx = j - 4;
                        if (max_score >= 0.5f)
                        {
                            break;
                        }
                    }
                }
                if (max_score >= conf)
                {
                    predictions.Add(
                        new(
                            new(max_score_idx,
                                Labels.ElementAt(max_score_idx).Key,
                                Labels.ElementAt(max_score_idx).Value),
                            new((input.ElementAt(col_len_caches[0] + i) - input.ElementAt(col_len_caches[2] + i) * 0.5f) * width_scale,
                                (input.ElementAt(col_len_caches[1] + i) - input.ElementAt(col_len_caches[3] + i) * 0.5f) * height_scale,
                                input.ElementAt(col_len_caches[2] + i) * width_scale, input.ElementAt(col_len_caches[3] + i) * height_scale),
                            max_score));
                }
            });
            return predictions;
        }

        public void ResizeImage(Image image)
        {
            graphics.DrawImage(image, 0, 0, Imgsz, Imgsz);
        }

        private List<YoloPrediction> Suppress(List<YoloPrediction> items, float iou_conf)
        {
            List<YoloPrediction> result = new(items);
            foreach (YoloPrediction item in items)
            {
                foreach (YoloPrediction current in result.ToList()) // make a copy for each iteration
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
    }
}
