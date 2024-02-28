using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;
using System.Drawing;
using YOLO.Extentions;
using YOLO.Models;


namespace YOLO
{
    public class Yolov8 : IDisposable
    {
        private readonly InferenceSession _inferenceSession;
        private readonly YoloModel _model = new();
        int N_Class { get; set; }
        int Imgsz { get; set; }

        public Yolov8(string modelPath, bool useCuda = false)
        {

            if (useCuda)
            {
                OrtCUDAProviderOptions cudaProviderOptions = new();
                cudaProviderOptions.UpdateOptions(new()
                {
                    { "cudnn_conv_use_max_workspace", "1" },
                    { "cudnn_conv1d_pad_to_nc1d", "1" },
                    { "arena_extend_strategy", "kSameAsRequested" },
                    { "do_copy_in_default_stream", "1" }
                });
                SessionOptions opts = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
                opts.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                _inferenceSession = new(modelPath, opts);
            }
            else
            {
                SessionOptions opts = new();
                _inferenceSession = new(modelPath, opts);
            }
            // Get model info
            get_input_details();
            get_output_details();
            using Bitmap bitmap = new(Imgsz, Imgsz);
            NamedOnnxValue[] inputs = [NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(bitmap))];
            _inferenceSession.Run(inputs, _model.Outputs);
        }

        public void SetupLabels(Dictionary<string, Color> color_mapper)
        {
            int i = 0;
            foreach (KeyValuePair<string, Color> keyValuePair in color_mapper)
            {
                _model.Labels.Add(new(i, keyValuePair.Key, keyValuePair.Value));
                ++i;
            }
        }

        public List<YoloPrediction> Predict(Bitmap image, float conf_thres = 0, float iou_thres = 0)
        {
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = Inference(image);
            return Suppress(ParseOutput(outputs, image, conf_thres), iou_thres);
        }

        /// <summary>
        /// Removes overlapped duplicates (nms).
        /// </summary>
        private List<YoloPrediction> Suppress(List<YoloPrediction> items, float iou_conf)
        {
            List<YoloPrediction> result = new(items);
            foreach (YoloPrediction item in items) // iterate every prediction
            {
                foreach (YoloPrediction current in items.ToList()) // make a copy for each iteration
                {
                    if (current != item)
                    {
                        float intArea = RectangleF.Intersect(item.Rectangle, current.Rectangle).Area();
                        if (intArea / (item.Area + current.Area - intArea) >= iou_conf)
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

        public YoloClassifyPrediction ClassifyPredict(Image img)
        {
            NamedOnnxValue[] inputs =
            [
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(Utils.ResizeImage(img, Imgsz, Imgsz)))
            ];
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = _inferenceSession.Run(inputs);

            List<DenseTensor<float>> output = [];

            foreach (string item in _model.Outputs) // add outputs for processing
            {
                output.Add(result.First(x => x.Name == item).Value as DenseTensor<float>);
            }

            float[] index_prob = Argmax_Score(output[0]);
            return new YoloClassifyPrediction(_model.Labels[(int)index_prob[0]], index_prob[1]);
        }

        private float[] Argmax_Score(DenseTensor<float> probs)
        {
            int max_index = 0;
            float max_prob = 0.0f;
            int i = 0;
            foreach (float prob in probs)
            {
                if (prob > max_prob)
                {
                    max_prob = prob;
                    max_index = i;
                }
                ++i;
                if (prob >= 0.5f)
                {
                    break;
                }
            }
            return [max_index, max_prob];
        }

        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Inference(Image img)
        {
            NamedOnnxValue[] inputs =
            [
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(Utils.ResizeImage(img, Imgsz, Imgsz)))
            ];

            return _inferenceSession.Run(inputs, _model.Outputs);
        }

        private List<YoloPrediction> ParseOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, Image image, float conf)
        {
            string firstOutput = _model.Outputs[0];
            DenseTensor<float> output = (DenseTensor<float>)outputs.First(x => x.Name == firstOutput).Value;
            return ParseDetect(output, image, conf);
        }

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image, float conf)
        {
            ConcurrentBag<YoloPrediction> result = new();

            var (w, h) = ((float)image.Width, (float)image.Height);
            var (xGain, yGain) = (Imgsz / w, Imgsz / h);
            float gain = Math.Min(xGain, yGain);
            float gain_inv = 1.0f / gain;
            var (xPad, yPad) = ((Imgsz - w * gain) * 0.5f, (Imgsz - h * gain) * 0.5f);

            Parallel.For(0, output.Dimensions[0], i =>
            {
                Parallel.For(0, (int)(output.Length / output.Dimensions[1]), j =>
                {
                    int dim = output.Strides[1];
                    Span<float> span = output.Buffer.Span[(i * output.Strides[0])..];

                    float a = span[j];
                    float b = span[dim + j];
                    float c = span[2 * dim + j];
                    float d = span[3 * dim + j];
                    float x_min = a - c * 0.5f - xPad;
                    float y_min = b - d * 0.5f - yPad;
                    float width = (a + c * 0.5f - xPad - x_min) * gain_inv;
                    float height = (b + d * 0.5f - yPad - y_min) * gain_inv;

                    for (int l = 0; l < _model.Dimensions - 4; l++)
                    {
                        float pred = span[(4 + l) * dim + j];

                        if (pred >= conf)
                        {
                            result.Add(new(_model.Labels[l], new(x_min * gain_inv, y_min * gain_inv, width, height), pred));
                        }
                    }
                });
            });
            return [.. result];
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
    }
}
