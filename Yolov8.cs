using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using System.Collections.Concurrent;
using System.Drawing;
using YOLO.Extentions;
using YOLO.Models;

namespace YOLO
{
    public class Yolov8 : Yolo, IDisposable
    {
        private readonly InferenceSession _inferenceSession;
        private readonly YoloModel _model = new YoloModel();
        private bool _useNumpy;
        int Imgsz { get; set; }

        public Yolov8(string modelPath, bool useCuda = false)
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

        public List<YoloPrediction> Predict(Image image, float conf_thres = 0, float iou_thres = 0, bool useNumpy = false)
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

            _useNumpy = useNumpy;
            using var outputs = Inference(image);
            return Suppress(ParseOutput(outputs, image));
        }

        /// <summary>
        /// Removes overlapped duplicates (nms).
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

        public YoloClassifyPrediction ClassifyPredict(Image img)
        {
            Bitmap resized;

            if (img.Width != _model.Width || img.Height != _model.Height)
            {
                resized = Utils.ResizeImage(img, _model.Width, _model.Height);
            }
            else
            {
                resized = new(img);
            }

            List<NamedOnnxValue> inputs = new()
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels(resized))
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
            Bitmap resized;

            if (img.Width != _model.Width || img.Height != _model.Height)
            {
                resized = Utils.ResizeImage(img, _model.Width, _model.Height);
            }
            else
            {
                resized = new(img);
            }

            var inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(resized))
            };

            return _inferenceSession.Run(inputs, _model.Outputs);
        }

        private List<YoloPrediction> ParseOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, Image image)
        {
            string firstOutput = _model.Outputs[0];
            var output = (DenseTensor<float>)outputs.First(x => x.Name == firstOutput).Value;

            if (_useNumpy)
            {
                return ParseDetectNumpy(output, image);
            }

            return ParseDetect(output, image);
        }

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image)
        {
            var result = new ConcurrentBag<YoloPrediction>();

            var (w, h) = (image.Width, image.Height);
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h);
            var gain = Math.Min(xGain, yGain);

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
                        var pred = span[(4 + l) * dim + j];

                        if (pred < _model.Confidence) continue;
                        var label = _model.Labels[l];
                        result.Add(new YoloPrediction
                        {
                            Label = label,
                            Score = pred,
                            Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                        });
                    }
                });
            });

            return result.ToList();
        }


        private List<YoloPrediction> ParseDetectNumpy(DenseTensor<float> output, Image image)
        {
            float[] outputArray = output.ToArray();
            var numpyArray = np.array(outputArray, np.float32);
            var data = numpyArray.reshape(84, 8400).transpose(new int[] { 1, 0 });
            return ProcessResult(data, image);
        }

        private List<YoloPrediction> ProcessResult(NDArray data, Image image)
        {
            var result = new ConcurrentBag<YoloPrediction>();
            var scores = np.max(data[":, 4:"], axis: 1);

            var temp = data[scores > 0.2f];
            scores = scores[scores > 0.2f];
            var class_ids = np.argmax(temp[":, 4:"], 1);
            var boxes = extract_rect(temp, image.Width, image.Height);
            var indices = nms(boxes, scores);
            foreach (var x in indices)
            {
                var label = _model.Labels[class_ids[x]];
                var prediction = new YoloPrediction(label, scores[x])
                {
                    Rectangle = new RectangleF(boxes[x][0], boxes[x][1], boxes[x][2] - boxes[x][0], boxes[x][3] - boxes[x][1])
                };
                result.Add(prediction);
            };
            return result.ToList();
        }


        private int[] nms(NDArray boxes, NDArray scores, float iou_threshold = .5f)
        {

            // Sort by score
            var sortedIndices = np.argsort<float>(scores)["::-1"];

            List<int> keepBoxes = new List<int>();
            int[] sortedIndicesArray = sortedIndices.Data<int>().ToArray();
            while (sortedIndicesArray.Length > 0)
            {
                // Pick the last box
                int boxId = sortedIndicesArray[0];
                keepBoxes.Add(boxId);
                // Compute IoU of the picked box with the rest
                NDArray ious = ComputeIOU(boxes[boxId], boxes[sortedIndices["1:"]]);

                // Remove boxes with IoU over the threshold
                var keepIndices = ious.Data<float>().AsQueryable().ToArray().Select(x => x < iou_threshold).ToArray();
                sortedIndicesArray = sortedIndicesArray.Skip(keepIndices.Length + 1).ToArray();
            }

            return keepBoxes.ToArray();
        }

        private NDArray ComputeIOU(NDArray box, NDArray boxes)
        {
            // Compute xmin, ymin, xmax, ymax for both boxes
            var xmin = np.maximum(box[0], boxes[":", 0]);
            var ymin = np.maximum(box[1], boxes[":", 1]);
            var xmax = np.minimum(box[2], boxes[":", 2]);
            var ymax = np.minimum(box[3], boxes[":", 3]);

            // Compute intersection area
            var intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin);

            // Compute union area
            var box_area = (box[2] - box[0]) * (box[3] - box[1]);
            var boxes_area = (boxes[":", 2] - boxes[":", 0]) * (boxes[":", 3] - boxes[":", 1]);
            var union_area = box_area + boxes_area - intersection_area;

            // Compute IoU
            var iou = intersection_area / union_area;

            return iou;
        }

        private NDArray extract_rect(NDArray temp, int width, int height)
        {
            var data = rescale_boxes(temp[":, :4"], width, height);
            var boxes = Xywh2Xyxy(data);
            return boxes;
        }

        public NDArray Xywh2Xyxy(NDArray x)
        {
            var y = x.Clone();
            y[":", 0] = x[":", 0] - x[":", 2] / 2;
            y[":", 1] = x[":", 1] - x[":", 3] / 2;
            y[":", 2] = x[":", 0] + x[":", 2] / 2;
            y[":", 3] = x[":", 1] + x[":", 3] / 2;
            return y;
        }

        private NDArray rescale_boxes(NDArray boxes, int width, int height)
        {

            NDArray inputShape = np.array(new float[] { _model.Width, _model.Height, _model.Width, _model.Height });
            NDArray resizedBoxes = np.divide(boxes, inputShape);
            resizedBoxes = np.multiply(resizedBoxes, new float[] { width, height, width, height });
            return resizedBoxes;
        }

        private void prepare_input(Image img)
        {
            Bitmap bmp = Utils.ResizeImage(img, _model.Width, _model.Height);

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
