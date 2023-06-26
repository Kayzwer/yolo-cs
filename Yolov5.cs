﻿using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;
using System.Drawing;
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

        public Yolov5(string modelPath, bool useCuda = false)
        {

            if (useCuda)
            {
                SessionOptions opts = SessionOptions.MakeSessionOptionWithCudaProvider();
                opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
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

        public List<YoloPrediction> Predict(Image image, float conf_thres = 0, float iou_thres = 0)
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
            return Suppress(ParseOutput(outputs, image));
        }

        public YoloClassifyPrediction ClassifyPredict(Image img)
        {
            Bitmap resized;

            if (img.Width != _model.Width || img.Height != _model.Height)
            {
                resized = Utils.ResizeImage(img, _model.Width, _model.Height); // fit image size to specified input size
            }
            else
            {
                resized = new(img);
            }

            List<NamedOnnxValue> inputs = new() // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels(resized))
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
        private List<YoloPrediction> Suppress(List<YoloPrediction> items)
        {
            var result = new List<YoloPrediction>(items);

            foreach (var item in items) // iterate every prediction
            {
                foreach (var current in result.ToList()) // make a copy for each iteration
                {
                    if (current == item) continue;

                    var (rect1, rect2) = (item.Rectangle, current.Rectangle);

                    var intersection = RectangleF.Intersect(rect1, rect2);

                    float intArea = intersection.Area(); // intersection area
                    float unionArea = rect1.Area() + rect2.Area() - intArea; // union area
                    float overlap = intArea / unionArea; // overlap ratio

                    if (overlap >= _model.Overlap)
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
            Bitmap resized;

            if (img.Width != _model.Width || img.Height != _model.Height)
            {
                resized = Utils.ResizeImage(img, _model.Width, _model.Height); // fit image size to specified input size
            }
            else
            {
                resized = img as Bitmap ?? new Bitmap(img);
            }

            var inputs = new[] // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(resized))
            };

            return _inferenceSession.Run(inputs, _model.Outputs); // run inference
        }

        private List<YoloPrediction> ParseOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, Image image)
        {
            if (_model.UseDetect)
            {
                string firstOutput = _model.Outputs[0];
                var output = (DenseTensor<float>)outputs.First(x => x.Name == firstOutput).Value;
                return ParseDetect(output, image);
            }

            return ParseSigmoid(outputs, image);
        }

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image)
        {
            var result = new ConcurrentBag<YoloPrediction>();

            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h); // x, y gains
            var gain = Math.Min(xGain, yGain); // gain = resized / original

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2); // left, right pads

            Parallel.For(0, (int)output.Length / _model.Dimensions, i =>
            {
                var span = output.Buffer.Span.Slice(i * _model.Dimensions);
                if (span[4] <= _model.Confidence) return; // skip low obj_conf results

                for (int j = 5; j < _model.Dimensions; j++)
                {
                    span[j] *= span[4]; // mul_conf = obj_conf * cls_conf
                }

                float xMin = (span[0] - span[2] / 2 - xPad) / gain; // unpad bbox tlx to original
                float yMin = (span[1] - span[3] / 2 - yPad) / gain; // unpad bbox tly to original
                float xMax = (span[0] + span[2] / 2 - xPad) / gain; // unpad bbox brx to original
                float yMax = (span[1] + span[3] / 2 - yPad) / gain; // unpad bbox bry to original

                xMin = Utils.Clamp(xMin, 0, w - 0); // clip bbox tlx to boundaries
                yMin = Utils.Clamp(yMin, 0, h - 0); // clip bbox tly to boundaries
                xMax = Utils.Clamp(xMax, 0, w - 1); // clip bbox brx to boundaries
                yMax = Utils.Clamp(yMax, 0, h - 1); // clip bbox bry to boundaries

                for (int k = 5; k < _model.Dimensions; k++)
                {
                    if (span[k] <= _model.MulConfidence) continue; // skip low mul_conf results

                    var label = _model.Labels[k - 5];
                    var prediction = new YoloPrediction(label, span[k])
                    {
                        Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                    };

                    result.Add(prediction);
                }
            });

            return result.ToList();
        }

        private List<YoloPrediction> ParseSigmoid(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, Image image)
        {
            return new List<YoloPrediction>();
        }

        private void prepare_input(Image img)
        {
            var bmp = Utils.ResizeImage(img, _model.Width, _model.Height);
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
            List<YoloPrediction> predictions = Predict(clone, conf_thres, iou_thres);
            foreach (YoloPrediction prediction in predictions)
            {
                if (class_conf[prediction.Label.Name] > prediction.Score)
                {
                    predictions.Remove(prediction);
                }
            }
            return predictions;
        }
    }
}
