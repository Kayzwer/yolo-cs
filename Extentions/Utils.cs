using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

namespace YOLO.Extentions
{
    public static class Utils
    {
        public static Bitmap ResizeImage(Image image, int target_width, int target_height)
        {
            PixelFormat format = image.PixelFormat;

            var output = new Bitmap(target_width, target_height, format);

            var (w, h) = ((float)image.Width, (float)image.Height); // image width and height
            var (xRatio, yRatio) = (target_width / w, target_height / h); // x, y ratios
            var ratio = Math.Min(xRatio, yRatio); // ratio = resized / original
            var (width, height) = ((int)(w * ratio), (int)(h * ratio)); // roi width and height
            var (x, y) = ((int)((target_width * 0.5f) - (width * 0.5f)), (int)((target_height * 0.5f) - (height * 0.5f))); // roi x and y coordinates
            var roi = new Rectangle(x, y, width, height); // region of interest

            using (var graphics = Graphics.FromImage(output))
            {
                graphics.Clear(Color.FromArgb(0, 0, 0, 0)); // clear canvas

                graphics.SmoothingMode = SmoothingMode.None; // no smoothing
                graphics.InterpolationMode = InterpolationMode.NearestNeighbor; // nn interpolation
                graphics.PixelOffsetMode = PixelOffsetMode.Half; // half pixel offset

                graphics.DrawImage(image, roi); // draw scaled
            }

            return output;
        }
        //https://github.com/ivilson/Yolov7net/issues/17
        public static Tensor<float> ExtractPixels2(Bitmap bitmap)
        {
            int pixelCount = bitmap.Width * bitmap.Height;
            Rectangle rectangle = new(0, 0, bitmap.Width, bitmap.Height);
            DenseTensor<float> tensor = new(new[] { 1, 3, bitmap.Height, bitmap.Width });
            Span<byte> data;

            BitmapData bitmapData;
            if (bitmap.PixelFormat == PixelFormat.Format24bppRgb && bitmap.Width % 4 == 0)
            {
                bitmapData = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

                unsafe
                {
                    data = new Span<byte>((void*)bitmapData.Scan0, bitmapData.Height * bitmapData.Stride);
                }

                ExtractPixelsRgb(tensor, data, pixelCount);
            }
            else
            {
                // force convert to 32 bit PArgb
                bitmapData = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);

                unsafe
                {
                    data = new Span<byte>((void*)bitmapData.Scan0, bitmapData.Height * bitmapData.Stride);
                }

                ExtractPixelsArgb(tensor, data, pixelCount);
            }

            bitmap.UnlockBits(bitmapData);

            return tensor;
        }

        public static void ExtractPixelsArgb(DenseTensor<float> tensor, Span<byte> data, int pixelCount)
        {
            Span<float> spanR = tensor.Buffer.Span;
            Span<float> spanG = spanR[pixelCount..];
            Span<float> spanB = spanG[pixelCount..];

            int sidx = 0;
            for (int i = 0; i < pixelCount; i++)
            {
                spanR[i] = data[sidx + 2] * 0.0039215686274509803921568627451f;
                spanG[i] = data[sidx + 1] * 0.0039215686274509803921568627451f;
                spanB[i] = data[sidx] * 0.0039215686274509803921568627451f;
                sidx += 4;
            }
        }

        public static void ExtractPixelsRgb(DenseTensor<float> tensor, Span<byte> data, int pixelCount)
        {
            Span<float> spanR = tensor.Buffer.Span;
            Span<float> spanG = spanR[pixelCount..];
            Span<float> spanB = spanG[pixelCount..];

            int sidx = 0;
            for (int i = 0; i < pixelCount; i++)
            {
                spanR[i] = data[sidx + 2] * 0.0039215686274509803921568627451f;
                spanG[i] = data[sidx + 1] * 0.0039215686274509803921568627451f;
                spanB[i] = data[sidx] * 0.0039215686274509803921568627451f;
                sidx += 3;
            }
        }

        public static float Clamp(float value, float min, float max)
        {
            return value < min ? min : value > max ? max : value;
        }

        public static Image DrawBoundingBox(Image image, List<YoloPrediction> predictions, int bounding_box_thickness, int font_size)
        {
            using Graphics graphics = Graphics.FromImage(image);
            foreach (YoloPrediction prediction in predictions) // iterate predictions to draw results
            {
                double score = Math.Round(prediction.Score, 2);
                graphics.DrawRectangles(new Pen(prediction.Label.Color, bounding_box_thickness), new[] { prediction.Rectangle });
                var (x, y) = (prediction.Rectangle.X, prediction.Rectangle.Y);
                graphics.DrawString($"{prediction.Label.Name} ({score})",
                                new Font("Consolas", font_size, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),
                                new PointF(x, y));
            }
            return image;
        }
    }
}
