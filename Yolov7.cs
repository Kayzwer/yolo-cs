using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;
using YOLO.Extentions;
using YOLO.Models;


namespace YOLO
{
    public class Yolov7 : Yolo, IDisposable
    {
        private readonly InferenceSession _inferenceSession;
        private readonly YoloModel _model = new YoloModel();
        int Imgsz { get; set; }
        int N_Class { get; set; }

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
            using Bitmap bitmap = new(Imgsz, Imgsz);
            NamedOnnxValue[] inputs = { NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(bitmap)) };
            _inferenceSession.Run(inputs, _model.Outputs);
        }

        public override void SetupLabels(Dictionary<string, Color> color_mapper)
        {
            int i = 0;
            foreach (KeyValuePair<string, Color> keyValuePair in color_mapper)
            {
                _model.Labels.Add(new(i, keyValuePair.Key, keyValuePair.Value));
                ++i;
            }
        }

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image, Dictionary<string, float> class_conf, float conf)
        {
            ConcurrentBag<YoloPrediction> result = new();
            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (Imgsz / (float)w, Imgsz / (float)h); // x, y gains
            float gain = Math.Min(xGain, yGain); // gain = resized / original
            float gain_inv = 1.0f / gain;
            var (xPad, yPad) = ((Imgsz - w * gain) * 0.5f, (Imgsz - h * gain) * 0.5f); // left, right pads

            Parallel.For(0, output.Dimensions[0], i =>
            {
                Span<float> span = output.Buffer.Span[(i * output.Strides[0])..];
                YoloLabel label = _model.Labels[(int)span[5]];
                if (span[6] >= class_conf[label.Name] && span[6] >= conf)
                {
                    float xMin = (span[1] - xPad) * gain_inv;
                    float yMin = (span[2] - yPad) * gain_inv;
                    float xMax = (span[3] - xPad) * gain_inv;
                    float yMax = (span[4] - yPad) * gain_inv;
                    result.Add(new(label, new(xMin, yMin, xMax - xMin, yMax - yMin), span[6]));
                }
            });
            return result.ToList();
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
                    if ((intArea / (item.Area + current.Area - intArea)) >= iou_conf)
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

        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Inference(Image img)
        {
            NamedOnnxValue[] inputs = new[] // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(Utils.ResizeImage(img, Imgsz, Imgsz)))
            };

            return _inferenceSession.Run(inputs, _model.Outputs); // run inference
        }

        private void get_input_details()
        {
            Imgsz = _inferenceSession.InputMetadata["images"].Dimensions[2];
        }

        private void get_output_details()
        {
            _model.Outputs = _inferenceSession.OutputMetadata.Keys.ToArray();
            _model.Dimensions = _inferenceSession.OutputMetadata[_model.Outputs[0]].Dimensions[1];
            N_Class = _inferenceSession.OutputMetadata.ToArray()[0].Value.Dimensions[1] - 4;
        }

        public void Dispose()
        {
            _inferenceSession.Dispose();
        }

        public override List<YoloPrediction> Predict(Bitmap img, Dictionary<string, float> class_conf, float conf_thres = 0, float iou_thres = 0)
        {
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = Inference(img);
            string firstOutput = _model.Outputs[0];
            DenseTensor<float> output = (DenseTensor<float>)outputs.First(x => x.Name == firstOutput).Value;
            return Suppress(ParseDetect(output, img, class_conf, conf_thres), iou_thres);
        }

        public override int GetModelNClass()
        {
            return N_Class;
        }
    }
}
