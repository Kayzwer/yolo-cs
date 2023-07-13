using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;
using System.Drawing.Drawing2D;
using YOLO.Extentions;
using YOLO.Models;

namespace YOLO
{
    public class Yolov8 : Yolo, IDisposable
    {
        private readonly InferenceSession _inferenceSession;
        private readonly YoloModel _model = new();
        int Imgsz { get; set; }
        Bitmap resized_img { get; set; }
        Graphics graphics { get; set; }

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

        public override void SetupLabels(Dictionary<string, Color> color_mapper)
        {
            int i = 0;
            foreach (KeyValuePair<string, Color> keyValuePair in color_mapper)
            {
                _model.Labels.Add(new(i, keyValuePair.Key, keyValuePair.Value));
                ++i;
            }
        }

        public List<YoloPrediction> Predict(Image image, Dictionary<string, float> class_conf, float conf_thres = 0, float iou_thres = 0)
        {
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = Inference(image);
            return Suppress(ParseOutput(outputs, image, class_conf), iou_thres);
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
            ResizeImage(img);

            List<NamedOnnxValue> inputs = new()
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels(resized_img))
            };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = _inferenceSession.Run(inputs);

            List<DenseTensor<float>> output = new();


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
            return new float[] { max_index, max_prob };
        }

        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Inference(Image img)
        {
            ResizeImage(img);

            NamedOnnxValue[] inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(resized_img))
            };

            return _inferenceSession.Run(inputs, _model.Outputs);
        }

        private List<YoloPrediction> ParseOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, Image image, Dictionary<string, float> class_conf)
        {
            string firstOutput = _model.Outputs[0];
            DenseTensor<float> output = (DenseTensor<float>)outputs.First(x => x.Name == firstOutput).Value;
            return ParseDetect(output, image, class_conf);
        }

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image, Dictionary<string, float> class_conf)
        {
            ConcurrentBag<YoloPrediction> result = new();

            var (w, h) = (image.Width, image.Height);
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h);
            float gain = Math.Min(xGain, yGain);

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2);

            Parallel.For(0, output.Dimensions[0], i =>
            {
                Parallel.For(0, (int)(output.Length / output.Dimensions[1]), j =>
                {
                    int dim = output.Strides[1];
                    var span = output.Buffer.Span[(i * output.Strides[0])..];

                    float a = span[j];
                    float b = span[dim + j];
                    float c = span[2 * dim + j];
                    float d = span[3 * dim + j];
                    float xMin = (a - c / 2 - xPad) / gain; // unpad bbox tlx to original
                    float yMin = (b - d / 2 - yPad) / gain; // unpad bbox tly to original
                    float xMax = (a + c / 2 - xPad) / gain; // unpad bbox brx to original
                    float yMax = (b + d / 2 - yPad) / gain; // unpad bbox bry to original

                    for (int l = 0; l < _model.Dimensions - 4; l++)
                    {
                        float pred = span[(4 + l) * dim + j];

                        if (pred < _model.Confidence || pred < class_conf[_model.Labels[l].Name]) continue;
                        YoloLabel label = _model.Labels[l];
                        result.Add(new(label, new(xMin, yMin, xMax - xMin, yMax - yMin), pred));
                    }
                });
            });

            return result.ToList();
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
            return Predict(clone, class_conf, conf_thres, iou_thres);
        }
    }
}
