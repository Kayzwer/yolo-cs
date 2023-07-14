using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;
using System.Drawing.Drawing2D;
using YOLO.Extentions;
using YOLO.Models;

namespace YOLO
{
    /// <summary>
    /// yolov5、yolov6 模型,不包含nms结果
    /// </summary>
    public class Yolov5 : Yolo, IDisposable
    {
        private readonly InferenceSession _inferenceSession;
        private readonly YoloModel _model = new YoloModel();
        int Imgsz { get; set; }
        Bitmap resized_img { get; set; }
        Graphics graphics { get; set; }

        public Yolov5(string modelPath, bool useCuda = false)
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
            if (conf_thres > 0f)
            {
                _model.Confidence = conf_thres;
                _model.MulConfidence = conf_thres + 0.05f;
            }

            if (iou_thres > 0f)
            {
                _model.Overlap = iou_thres;
            }

            using var outputs = Inference(image);
            return Suppress(ParseOutput(outputs, image, class_conf), iou_thres);
        }

        public YoloClassifyPrediction ClassifyPredict(Image img)
        {
            ResizeImage(img);

            List<NamedOnnxValue> inputs = new() // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(resized_img))
            };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = _inferenceSession.Run(inputs); // run inference

            List<DenseTensor<float>> output = new();

            foreach (var item in _model.Outputs) // add outputs for processing
            {
                output.Add(result.First(x => x.Name == item).Value as DenseTensor<float>);
            }

            float[] index_prob = Argmax_Score(Softmax(output[0].ToArray()));
            return new YoloClassifyPrediction(_model.Labels[(int)index_prob[0]], index_prob[1]);
        }

        private float[] Argmax_Score(float[] probs)
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

        private float[] Softmax(float[] input)
        {
            float[] prob_vec = new float[input.Length];
            float deno = 0.0f;
            foreach (float num in input)
            {
                deno += (float)Math.Exp(num);
            }

            for (int i = 0; i < prob_vec.Length; ++i)
            {
                prob_vec[i] = (float)Math.Exp(input[i]) / deno;
            }

            return prob_vec;
        }

        /// <summary>
        /// Removes overlaped duplicates (nms).
        /// </summary>
        private List<YoloPrediction> Suppress(List<YoloPrediction> items, float iou_conf)
        {
            List<YoloPrediction> result = new(items);
            foreach (YoloPrediction item in items) // iterate every prediction
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

        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Inference(Image img)
        {
            ResizeImage(img);

            var inputs = new[] // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(resized_img))
            };

            return _inferenceSession.Run(inputs, _model.Outputs); // run inference
        }

        private List<YoloPrediction> ParseOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, Image image, Dictionary<string, float> class_conf)
        {
            if (_model.UseDetect)
            {
                string firstOutput = _model.Outputs[0];
                var output = (DenseTensor<float>)outputs.First(x => x.Name == firstOutput).Value;
                return ParseDetect(output, image, class_conf);
            }

            return ParseSigmoid(outputs, image);
        }

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image, Dictionary<string, float> class_conf)
        {
            var result = new ConcurrentBag<YoloPrediction>();

            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h); // x, y gains
            var gain = Math.Min(xGain, yGain); // gain = resized / original

            var (xPad, yPad) = ((_model.Width - w * gain) * 0.5f, (_model.Height - h * gain) * 0.5f); // left, right pads

            Parallel.For(0, (int)output.Length / _model.Dimensions, i =>
            {
                var span = output.Buffer.Span[(i * _model.Dimensions)..];
                if (span[4] <= _model.Confidence) return; // skip low obj_conf results

                for (int j = 5; j < _model.Dimensions; j++)
                {
                    span[j] *= span[4]; // mul_conf = obj_conf * cls_conf
                }

                float xMin = (span[0] - span[2] * 0.5f - xPad) / gain; // unpad bbox tlx to original
                float yMin = (span[1] - span[3] * 0.5f - yPad) / gain; // unpad bbox tly to original
                float xMax = (span[0] + span[2] * 0.5f - xPad) / gain; // unpad bbox brx to original
                float yMax = (span[1] + span[3] * 0.5f - yPad) / gain; // unpad bbox bry to original

                xMin = Utils.Clamp(xMin, 0, w - 0); // clip bbox tlx to boundaries
                yMin = Utils.Clamp(yMin, 0, h - 0); // clip bbox tly to boundaries
                xMax = Utils.Clamp(xMax, 0, w - 1); // clip bbox brx to boundaries
                yMax = Utils.Clamp(yMax, 0, h - 1); // clip bbox bry to boundaries

                for (int k = 5; k < _model.Dimensions; k++)
                {
                    if (span[k] < _model.MulConfidence || span[k] < class_conf[_model.Labels[k - 5].Name]) continue; // skip low mul_conf results
                    result.Add(new(_model.Labels[k - 5], new(xMin, yMin, xMax - xMin, yMax - yMin), span[k]));
                }
            });

            return result.ToList();
        }

        private List<YoloPrediction> ParseSigmoid(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, Image image)
        {
            return new List<YoloPrediction>();
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
            _model.Dimensions = _inferenceSession.OutputMetadata[_model.Outputs[0]].Dimensions[2];
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
