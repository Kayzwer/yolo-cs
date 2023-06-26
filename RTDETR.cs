using Microsoft.ML.OnnxRuntime;
using System.Drawing.Drawing2D;
using YOLO.Extentions;

namespace YOLO
{
    public class RTDETR : Yolo
    {
        const int MAX_POSSIBLE_OBJECT = 300;

        InferenceSession InferenceSession { get; set; }
        string[] OutputData { get; set; }
        int Imgsz { get; set; }
        int N_Class { get; set; }
        Dictionary<string, Color> Labels { get; set; }
        Dictionary<string, float[,]> bboxes_n_scores { get; set; }
        float[] max_scores { get; set; }
        int[] max_classes { get; set; }
        bool[] is_over_conf { get; set; }
        Bitmap resized_img { get; set; }
        Graphics graphics { get; set; }
        int col_len { get; set; }
        float[] max_argmax_cache { get; set; }
        public RTDETR(string model_path, bool use_cuda)
        {
            if (use_cuda)
            {
                InferenceSession = new(model_path, SessionOptions.MakeSessionOptionWithCudaProvider());
            }
            else
            {
                InferenceSession = new(model_path);
            }
            Imgsz = InferenceSession.InputMetadata["images"].Dimensions[2];
            OutputData = InferenceSession.OutputMetadata.Keys.ToArray();
            resized_img = new(Imgsz, Imgsz);
            graphics = Graphics.FromImage(resized_img);
            graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
            max_argmax_cache = new float[2];

            using Bitmap bitmap = new(Imgsz, Imgsz);
            NamedOnnxValue[] inputs = { NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(bitmap)) };
            InferenceSession.Run(inputs, OutputData);
        }

        public override void SetupLabels(Dictionary<string, Color> labels)
        {
            N_Class = labels.Count;
            Labels = labels;

            // Reserve arrays space
            bboxes_n_scores = new()
            {
                { "bboxes", new float[MAX_POSSIBLE_OBJECT, 4] },
                { "scores", new float[MAX_POSSIBLE_OBJECT, N_Class] }
            };
            max_scores = new float[MAX_POSSIBLE_OBJECT];
            max_classes = new int[MAX_POSSIBLE_OBJECT];
            is_over_conf = new bool[MAX_POSSIBLE_OBJECT];
            col_len = 4 + N_Class;
        }

        public override List<YoloPrediction> Predict(Bitmap image, float conf, float iou_conf = .45f)
        {
            ResizeImage(image);
            NamedOnnxValue[] inputs = { NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(resized_img)) };
            DisposableNamedOnnxValue[] result = InferenceSession.Run(inputs, OutputData).ToArray();
            GetBboxes_n_Scores(result[0]);
            Max_Argmax();
            Over_conf(conf);
            return Suppress(Final_output(image.Width, image.Height), iou_conf);
        }

        public override List<YoloPrediction> Predict(Bitmap image, Dictionary<string, float> class_conf, float conf = 0, float iou_conf = 0)
        {
            ResizeImage(image);
            NamedOnnxValue[] inputs = { NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(resized_img)) };
            DisposableNamedOnnxValue[] result = InferenceSession.Run(inputs, OutputData).ToArray();
            GetBboxes_n_Scores(result[0]);
            Max_Argmax();
            Over_conf(conf);
            return Suppress(Final_output(image.Width, image.Height, class_conf), iou_conf);
        }

        public void GetBboxes_n_Scores(DisposableNamedOnnxValue input)
        {
            int cur_col = 0;
            int cur_row = 0;
            foreach (float value in input.AsTensor<float>())
            {
                if (cur_col < 4)
                {
                    bboxes_n_scores["bboxes"][cur_row, cur_col] = value;
                }
                else
                {
                    bboxes_n_scores["scores"][cur_row, cur_col - 4] = value;
                }
                cur_col++;
                if (cur_col == col_len)
                {
                    cur_col = 0;
                    cur_row++;
                }
            }
        }

        public void Max_Argmax()
        {
            Parallel.For(0, MAX_POSSIBLE_OBJECT, i =>
            {
                Get_max_argmax(Split(bboxes_n_scores["scores"], i));
                max_scores[i] = max_argmax_cache[0];
                max_classes[i] = (int)max_argmax_cache[1];
            });
        }

        public float[] Split(float[,] input, int row_idx)
        {
            float[] output = new float[N_Class];
            Parallel.For(0, N_Class, i =>
            {
                output[i] = input[row_idx, i];
            });
            return output;
        }

        public void Get_max_argmax(float[] input)
        {
            float max = float.NegativeInfinity;
            float max_idx = 0;
            for (int i = 0; i < input.Length; i++)
            {
                if (input[i] > max)
                {
                    max = input[i];
                    max_idx = i;
                    if (max >= 0.5f)
                    {
                        break;
                    }
                }
            }
            max_argmax_cache[0] = max;
            max_argmax_cache[1] = max_idx;
        }

        public void Over_conf(float conf)
        {
            Parallel.For(0, MAX_POSSIBLE_OBJECT, i =>
            {
                is_over_conf[i] = max_scores[i] >= conf;
            });
        }

        public List<YoloPrediction> Final_output(int image_width, int image_height)
        {
            List<YoloPrediction> outputs = new();
            for (int i = 0; i < is_over_conf.Length; ++i)
            {
                if (is_over_conf[i])
                {
                    KeyValuePair<string, Color> label = Labels.ElementAt(max_classes[i]);
                    YoloPrediction prediction = new()
                    {
                        Label = new()
                        {
                            Id = max_classes[i],
                            Color = label.Value,
                            Name = label.Key
                        },
                        Rectangle = new RectangleF(
                            (bboxes_n_scores["bboxes"][i, 0] - bboxes_n_scores["bboxes"][i, 2] / 2f) * image_width,
                            (bboxes_n_scores["bboxes"][i, 1] - bboxes_n_scores["bboxes"][i, 3] / 2f) * image_height,
                            bboxes_n_scores["bboxes"][i, 2] * image_width,
                            bboxes_n_scores["bboxes"][i, 3] * image_height
                            ),
                        Score = max_scores[i]
                    };
                    outputs.Add(prediction);
                }
            }
            return outputs;
        }

        public List<YoloPrediction> Final_output(int image_width, int image_height, Dictionary<string, float> class_conf)
        {
            List<YoloPrediction> outputs = new();
            for (int i = 0; i < is_over_conf.Length; ++i)
            {
                if (is_over_conf[i] && max_scores[i] >= class_conf[Labels.ElementAt(max_classes[i]).Key])
                {
                    KeyValuePair<string, Color> label = Labels.ElementAt(max_classes[i]);
                    outputs.Add(new YoloPrediction()
                    {
                        Label = new()
                        {
                            Id = max_classes[i],
                            Color = label.Value,
                            Name = label.Key
                        },
                        Rectangle = new RectangleF(
                            (bboxes_n_scores["bboxes"][i, 0] - bboxes_n_scores["bboxes"][i, 2] / 2f) * image_width,
                            (bboxes_n_scores["bboxes"][i, 1] - bboxes_n_scores["bboxes"][i, 3] / 2f) * image_height,
                            bboxes_n_scores["bboxes"][i, 2] * image_width,
                            bboxes_n_scores["bboxes"][i, 3] * image_height
                            ),
                        Score = max_scores[i]
                    });
                }
            }
            return outputs;
        }

        public void ResizeImage(Image image)
        {
            graphics.DrawImage(image, 0, 0, Imgsz, Imgsz);
        }

        public override List<YoloPrediction> Predict(Bitmap clone)
        {
            return Predict(clone, .25f);
        }

        private List<YoloPrediction> Suppress(List<YoloPrediction> items, float iou_conf)
        {
            List<YoloPrediction> result = new(items);
            foreach (YoloPrediction item in items) // iterate every prediction
            {
                foreach (YoloPrediction current in result.ToList()) // make a copy for each iteration
                {
                    if (current == item) continue;
                    float intArea = RectangleF.Intersect(item.Rectangle, current.Rectangle).Area();
                    if ((intArea / (item.Rectangle.Area() + current.Rectangle.Area() - intArea)) >= iou_conf)
                    {
                        if (item.Score >= current.Score)
                        {
                            result.Remove(current);
                        }
                    }
                }
            }
            return result;
        }
    }
}
