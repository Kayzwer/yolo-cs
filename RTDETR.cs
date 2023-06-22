using Microsoft.ML.OnnxRuntime;
using System.Drawing;
using YOLO.Extentions;

namespace YOLO
{
    public class RTDETR
    {
        int MAX_POSSIBLE_OBJECT = 300;

        InferenceSession InferenceSession { get; set; }
        string[] OutputData { get; set; }
        int Imgsz { get; set; }
        int N_Class { get; set; }
        Dictionary<string, Color> Labels { get; set; }
        Dictionary<string, float[,]> bboxes_n_scores { get; set; }
        Dictionary<string, float[]> scores_cls { get; set; }
        bool[] is_over_conf { get; set; }

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

            // Warmup
            NamedOnnxValue[] inputs = { NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(new Bitmap(Imgsz, Imgsz))) };
            _ = InferenceSession.Run(inputs, OutputData);
        }

        public void SetupLabels(Dictionary<string, Color> labels)
        {
            N_Class = labels.Count;
            Labels = labels;

            // Reserve arrays space
            bboxes_n_scores = new()
            {
                { "bboxes", new float[MAX_POSSIBLE_OBJECT, 4] },
                { "scores", new float[MAX_POSSIBLE_OBJECT, N_Class] }
            };
            scores_cls = new()
            {
                { "scores", new float[MAX_POSSIBLE_OBJECT] },
                { "classes", new float[MAX_POSSIBLE_OBJECT] }
            };
            is_over_conf = new bool[MAX_POSSIBLE_OBJECT];
        }

        public Bitmap Predict(Image image, float conf, int font_size, int bounding_box_thickness)
        {
            Bitmap resized_img = Utils.ResizeImage(image, Imgsz, Imgsz);
            NamedOnnxValue[] inputs = { NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(resized_img)) };
            DisposableNamedOnnxValue[] result = InferenceSession.Run(inputs, OutputData).ToArray();
            GetBboxes_n_Scores(result[0]);
            XYWH2XYWH();
            Max_Argmax();
            Over_conf(conf);
            List<float[]> predictions = Final_output();
            using Graphics graphics = Graphics.FromImage(resized_img);
            foreach (float[] prediction in predictions)
            {
                KeyValuePair<string, Color> label = Labels.ElementAt((int)prediction[5]);
                graphics.DrawRectangle(new Pen(label.Value, bounding_box_thickness), new RectangleF(prediction[0], prediction[1], prediction[2], prediction[3]));
                graphics.DrawString($"{label.Key} ({Math.Round(prediction[4], 2)})",
                                new Font("Consolas", font_size, GraphicsUnit.Pixel), new SolidBrush(label.Value),
                                new PointF(prediction[0], prediction[1]));
            }
            return resized_img;
        }

        public void GetBboxes_n_Scores(DisposableNamedOnnxValue input)
        {
            int col_len = 4 + N_Class;
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

        public void XYWH2XYWH()
        {
            Parallel.For(0, MAX_POSSIBLE_OBJECT, i =>
            {
                bboxes_n_scores["bboxes"][i, 0] = bboxes_n_scores["bboxes"][i, 0] - bboxes_n_scores["bboxes"][i, 2] / 2f;
                bboxes_n_scores["bboxes"][i, 1] = bboxes_n_scores["bboxes"][i, 1] - bboxes_n_scores["bboxes"][i, 3] / 2f;
                bboxes_n_scores["bboxes"][i, 2] = bboxes_n_scores["bboxes"][i, 2];
                bboxes_n_scores["bboxes"][i, 3] = bboxes_n_scores["bboxes"][i, 3];
            });
        }

        public void Max_Argmax()
        {
            Parallel.For(0, MAX_POSSIBLE_OBJECT, i =>
            {
                float[] max_argmax = Get_max_argmax(Split(bboxes_n_scores["scores"], i));
                scores_cls["scores"][i] = max_argmax[0];
                scores_cls["classes"][i] = max_argmax[1];
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

        public float[] Get_max_argmax(float[] input)
        {
            float max = float.NegativeInfinity;
            float max_idx = 0;
            for (int i = 0; i < input.Length; i++)
            {
                if (input[i] > max)
                {
                    max = input[i];
                    max_idx = i;
                }
            }
            return new float[] { max, max_idx };
        }

        public void Over_conf(float conf)
        {
            Parallel.For(0, MAX_POSSIBLE_OBJECT, i =>
            {
                is_over_conf[i] = scores_cls["scores"][i] > conf;
            });
        }

        public List<float[]> Final_output()
        {
            List<float[]> outputs = new();
            Parallel.For(0, MAX_POSSIBLE_OBJECT, i =>
            {
                if (is_over_conf[i])
                {
                    outputs.Add(new float[] {
                    bboxes_n_scores["bboxes"][i, 0] * Imgsz,
                    bboxes_n_scores["bboxes"][i, 1] * Imgsz,
                    bboxes_n_scores["bboxes"][i, 2] * Imgsz,
                    bboxes_n_scores["bboxes"][i, 3] * Imgsz,
                    scores_cls["scores"][i],
                    scores_cls["classes"][i]
                });
                }
            });
            return outputs;
        }
    }
}
