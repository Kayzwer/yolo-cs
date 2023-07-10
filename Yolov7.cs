using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;
using System.Drawing.Drawing2D;
using YOLO.Extentions;
using YOLO.Models;


namespace YOLO
{
    public class Yolov7 : Yolo, IDisposable
    {
        private readonly InferenceSession _inferenceSession;
        private readonly YoloModel _model = new YoloModel();
        int Imgsz;
        Bitmap resized_img { get; set; }
        Graphics graphics { get; set; }

        public Yolov7(string modelPath, bool useCuda = false)
        {
            if (useCuda)
            {
                OrtCUDAProviderOptions cudaProviderOptions = new();
                cudaProviderOptions.UpdateOptions(new Dictionary<string, string>()
                {
                    { "cudnn_conv_use_max_workspace", "1" },
                    { "cudnn_conv1d_pad_to_nc1d", "1" },
                    { "arena_extend_strategy", "kSameAsRequested" },
                    { "do_copy_in_default_stream", "1" }
                });
                SessionOptions opts = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
                opts.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                _inferenceSession = new InferenceSession(modelPath, opts);
            }
            else
            {
                SessionOptions opts = new();
                _inferenceSession = new InferenceSession(modelPath, opts);
            }

            // Get model info
            get_input_details();
            get_output_details();
            resized_img = new(Imgsz, Imgsz);
            graphics = Graphics.FromImage(resized_img);
            graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
            using Bitmap bitmap = new(Imgsz, Imgsz);
            NamedOnnxValue[] inputs = { NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(bitmap)) };
            _inferenceSession.Run(inputs, _model.Outputs);
        }

        public void ResizeImage(Image image)
        {
            graphics.DrawImage(image, 0, 0, Imgsz, Imgsz);
        }

        public void SetupLabels(string[] labels)
        {
            labels.Select((s, i) => new { i, s }).ToList().ForEach(item =>
            {
                _model.Labels.Add(new YoloLabel { Id = item.i, Name = item.s });
            });
        }

        public override void SetupLabels(Dictionary<string, Color> color_mapper)
        {
            int i = 0;
            foreach (KeyValuePair<string, Color> keyValuePair in color_mapper)
            {
                _model.Labels.Add(new YoloLabel { Id = i, Name = keyValuePair.Key, Color = keyValuePair.Value });
                ++i;
            }
        }

        public void SetupYoloDefaultLabels()
        {
            var s = new string[] { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
            SetupLabels(s);
        }

        public List<YoloPrediction> Predict(Image image)
        {
            using var outputs = Inference(image);
            string firstOutput = _model.Outputs[0];
            var output = (DenseTensor<float>)outputs.First(x => x.Name == firstOutput).Value;
            return ParseDetect(output, image);
        }

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image)
        {
            var result = new ConcurrentBag<YoloPrediction>();

            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h); // x, y gains
            var gain = Math.Min(xGain, yGain); // gain = resized / original

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2); // left, right pads

            Parallel.For(0, output.Dimensions[0], i =>
            {
                var span = output.Buffer.Span[(i * output.Strides[0])..];
                var label = _model.Labels[(int)span[5]];
                var prediction = new YoloPrediction(label, span[6]);

                var xMin = (span[1] - xPad) / gain;
                var yMin = (span[2] - yPad) / gain;
                var xMax = (span[3] - xPad) / gain;
                var yMax = (span[4] - yPad) / gain;

                //install package TensorFlow.Net,SciSharp.TensorFlow.Redist 安装这两个包可以用numpy 进行计算
                //var box = np.array(item.GetValue(1), item.GetValue(2), item.GetValue(3), item.GetValue(4));
                //var tmp =  np.array(xPad, yPad,xPad, yPad) ;
                //box -= tmp;
                //box /= gain;

                prediction.Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin);
                result.Add(prediction);
            });

            return result.ToList();
        }

        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Inference(Image img)
        {
            ResizeImage(img);

            var inputs = new[] // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(resized_img))
            };

            return _inferenceSession.Run(inputs, _model.Outputs); // run inference
        }

        private void get_input_details()
        {
            Imgsz = _inferenceSession.InputMetadata["images"].Dimensions[2];
            _model.Height = Imgsz;
            _model.Width = Imgsz;
        }

        private void get_output_details()
        {
            _model.Outputs = _inferenceSession.OutputMetadata.Keys.ToArray();
            _model.Dimensions = _inferenceSession.OutputMetadata[_model.Outputs[0]].Dimensions[1];
            _model.UseDetect = !(_model.Outputs.Any(x => x == "score"));
        }

        public void Dispose()
        {
            _inferenceSession.Dispose();
        }

        public override List<YoloPrediction> Predict(Bitmap clone)
        {
            return Predict(clone);
        }

        public override List<YoloPrediction> Predict(Bitmap clone, float conf_thres = 0, float iou_thres = 0)
        {
            return Predict(clone, conf_thres, iou_thres);
        }

        public override List<YoloPrediction> Predict(Bitmap clone, Dictionary<string, float> class_conf, float conf_thres = 0, float iou_thres = 0)
        {
            List<YoloPrediction> predictions = Predict(clone, conf_thres, iou_thres);
            int i = 0;
            int n = predictions.Count;
            while (i < n)
            {
                if (predictions[i].Score < class_conf[predictions[i].Label.Name])
                {
                    predictions.RemoveAt(i--);
                    n--;
                }
                i++;
            }
            return predictions;
        }
    }
}
