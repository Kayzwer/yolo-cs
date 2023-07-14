using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing.Imaging;

namespace YOLO.Extentions
{
    public static class Utils
    {
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
